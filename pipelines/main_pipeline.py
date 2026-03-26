import pandas as pd
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)

MAX_RETRIES = 3

from langgraph.graph import StateGraph, END
from .state import AgentState
from agents.planner_agent import PlannerAgent
from agents.sql_agent import SQLAgent
from execution.sql_executor import execute_sql
from agents.visualization_agent import VisualizationAgent
from agents.summary_agent import SummaryAgent
from utils.cache import semantic_cache, cache as cache_manager, get_dataset_hash


def direct_cache_node(state: AgentState) -> AgentState:
    """Checks the direct cache for an exact match."""
    print("---CHECKING DIRECT CACHE---")
    dataset_hash = get_dataset_hash(state['data_path'])
    cache_key = f"query:{dataset_hash}:{state['query']}"
    cached_result = cache_manager.get(cache_key)
    
    if cached_result:
        print("---DIRECT CACHE HIT---")
        return {**state, **cached_result, "direct_cache_hit": True}
    else:
        print("---DIRECT CACHE MISS---")
        return {**state, "direct_cache_hit": False}


def route_after_direct_cache(state: AgentState) -> str:
    """Determines the next step after the direct cache check."""
    if state.get("direct_cache_hit"):
        return END
    else:
        return "semantic_cache"


def semantic_cache_node(state: AgentState) -> AgentState:
    """Checks the semantic cache for a similar query."""
    print("---CHECKING SEMANTIC CACHE---")
    cached_result = semantic_cache.search(state['query'])
    if cached_result:
        print("---SEMANTIC CACHE HIT---")
        logger.info(f"Retrieved chunks: {cached_result}")
        # Merge the cached result into the current state
        return {**state, **cached_result, "semantic_cache_hit": True}
    else:
        print("---SEMANTIC CACHE MISS---")
        return {**state, "semantic_cache_hit": False}


def route_after_semantic_cache(state: AgentState) -> str:
    """Determines the next step after the semantic cache check."""
    if state.get("semantic_cache_hit"):
        return END
    else:
        return "analysis_router"



def route_to_analysis(state: AgentState) -> str:
    """
    Determines whether to perform schema analysis or skip it.
    If table_info is already in the state, it means the analysis has been done.
    """
    print("---ROUTING TO ANALYSIS OR PLANNER---")
    if state.get("table_info"):
        print("table_info found in state. Skipping analysis.")
        return "planner"
    else:
        print("table_info not found. Proceeding with schema analysis.")
        return "schema_analyzer"


def should_retry_analysis(state: AgentState) -> str:
    """Determines whether to retry schema analysis or give up."""
    print("---CHECKING FOR SCHEMA ANALYSIS ERRORS---")
    retries = state.get('retries', 0)
    if state.get('error'):
        if retries < MAX_RETRIES:
            print(f"Schema Analysis Error detected. Retry attempt {retries + 1}")
            state['retries'] = retries + 1
            return "retry"
        else:
            print("Schema Analysis Error detected. Max retries reached. Aborting.")
            return "end"
    print("No schema analysis error. Proceeding.")
    state['retries'] = 0
    return "continue"


def schema_analysis_node(state: AgentState) -> AgentState:
    """Analyzes the data using a hybrid approach: statistical profiling and dynamic random sampling."""
    try:
        print("---ANALYZING SCHEMA (HYBRID PROFILING + SAMPLING)---")
        data_path = state['data_path']
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            return {**state, "error": f"Unsupported file type: {data_path}"}

        # --- Hybrid Analysis --- 
        profile_parts = []

        # 1. Data Profile (Statistical Summary)
        profile_parts.append("### Data Profile:")
        profile_parts.append(f"- Total Rows: {len(df)}")
        profile_parts.append(f"- Duplicate Rows: {df.duplicated().sum()}")
        profile_parts.append("\n#### Column Data Types:")
        profile_parts.append(df.dtypes.to_string())
        
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            profile_parts.append("\n#### Null Value Counts:")
            profile_parts.append(null_counts[null_counts > 0].to_string())

        numeric_df = df.select_dtypes(include='number')
        if not numeric_df.empty:
            profile_parts.append("\n#### Statistical Summary for Numerical Columns:")
            profile_parts.append(numeric_df.describe().to_string())

        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            profile_parts.append("\n#### Summary for Categorical Columns:")
            profile_parts.append(categorical_df.describe().to_string())

        # 2. Dynamic Random Sampling
        num_rows = len(df)
        if num_rows <= 20:
            sample_size = num_rows
        elif num_rows <= 1000:
            sample_size = 20
        elif num_rows <= 10000:
            sample_size = 35
        else:
            sample_size = 50

        profile_parts.append(f"\n### Random Sample of {sample_size} Rows:")
        # Use a fixed random_state for reproducibility within the session
        sample_df = df.sample(n=sample_size, random_state=42)
        profile_parts.append(sample_df.to_string())

        # Combine all parts into a single string
        full_profile = "\n".join(profile_parts)

        return {
            **state, 
            "table_info": full_profile, # Pass the combined profile and samples
            "example_rows": "", # Deprecate the old field
            "error": None,
            "retries": 0
        }
    except Exception as e:
        return {**state, "error": f"Error in Schema Analysis: {e}", "retries": state.get('retries', 0) + 1}

def planner_node(state: AgentState) -> AgentState:
    if state.get('error') and state.get('retries', 0) == 0: return state
    try:
        print("---PLANNING---")
        history = state.get('history', [])
        if state.get('error'):
            history.append({
                "error": state.get('error')
            })

        planner_agent = PlannerAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        plan_and_chart_type = planner_agent.create_plan(state['query'], history, state['table_name'], state['table_info'])
        
        if "error" in plan_and_chart_type:
             return {**state, "error": plan_and_chart_type["error"], "retries": state.get('retries', 0) + 1}
             
        return {**state, **plan_and_chart_type, "error": None, "history": history, "retries": 0}
    except Exception as e:
        return {**state, "error": f"Error in Planner: {e}", "retries": state.get('retries', 0) + 1}

def sql_generator_node(state: AgentState) -> AgentState:
    if state.get('error') and state.get('retries', 0) == 0: return state
    try:
        print("---GENERATING SQL---")
        # On retry, the error from the previous step is the feedback
        history = state.get('history', [])
        if state.get('error'):
            history.append({
                "sql_query": state.get('sql_query'),
                "error": state.get('error')
            })

        sql_agent = SQLAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        sql_query = sql_agent.generate_sql(state['table_name'], state['table_info'], state['plan'], history)
        return {**state, "sql_query": sql_query, "error": None, "history": history, "retries": 0}
    except Exception as e:
        return {**state, "error": f"Error in SQL Generator: {e}", "retries": state.get('retries', 0) + 1}

def code_executor_node(state: AgentState) -> AgentState:
    if state.get('error') and not state.get('sql_query'): return state
    try:
        print("---EXECUTING CODE---")
        execution_result = execute_sql(state['sql_query'], state['data_path'], state['table_name'], state['dataset_hash'])
        logger.info(f"Code execution result: {execution_result}")
        if "error" in execution_result:
            return {**state, "error": f"Error in Code Execution: {execution_result['error']}", "retries": state.get('retries', 0) + 1}
        # Clear error on success
        return {**state, "execution_result": execution_result, "error": None, "retries": 0}
    except Exception as e:
        return {**state, "error": f"Error in Code Executor: {e}", "retries": state.get('retries', 0) + 1}

def should_retry_sql(state: AgentState) -> str:
    """Determines whether to retry SQL generation or give up."""
    print("---CHECKING FOR SQL ERRORS---")
    retries = state.get('retries', 0)
    if state.get('error'):
        if retries < MAX_RETRIES:
            print(f"SQL Error detected. Retry attempt {retries + 1}")
            return "retry"
        else:
            print("SQL Error detected. Max retries reached. Aborting.")
            return "end"
    print("No SQL error. Proceeding.")
    return "continue"

def visualization_router_node(state: AgentState) -> AgentState:
    """Determines if the result is suitable for the requested chart type and overrides if not."""
    if state.get('error'): return state
    
    print("---ROUTING VISUALIZATION---")
    chart_type = state.get('chart_type', 'none').lower()
    execution_result = state.get('execution_result')

    if not execution_result or not execution_result.get('result'):
        # No result, so no chart. We can just let it flow to the summarizer.
        state['chart_type'] = 'none'
        print("No execution result, setting chart type to none.")
        return state

    df = pd.DataFrame(execution_result['result'])

    # If a 2D chart is requested with only 1 column of data, it's un-plottable.
    if chart_type in ['line', 'bar', 'scatter', 'pie', 'heatmap'] and df.shape[1] < 2:
        print(f"Cannot create a '{chart_type}' chart with only one column of data. Overriding to 'table'.")
        state['chart_type'] = 'table'
    
    # If the result is a single value, it should always be a table.
    if df.shape[0] == 1 and df.shape[1] == 1:
        print(f"Result is a single value. Overriding to 'table'.")
        state['chart_type'] = 'table'

    # If the data has no variance (all rows are identical), a chart is meaningless.
    if df.nunique().nunique() == 1 and df.nunique().iloc[0] == 1:
        print("Data has no variance. Overriding to 'table'.")
        state['chart_type'] = 'table'

    return state

def visualization_node(state: AgentState) -> AgentState:
    if state.get('error') and state.get('retries', 0) == 0: return state
    try:
        print("---GENERATING VISUALIZATION---")
        history = state.get('history', [])
        if state.get('error'):
            history.append({
                "error": state.get('error')
            })

        visualization_agent = VisualizationAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        visualization_output = visualization_agent.generate_visualization(
            state['execution_result'], 
            state['chart_type'], 
            state['dataset_hash'], 
            state['query'],
            history
        )
        
        if "error" in visualization_output:
            return {**state, "error": f"Error in Visualization: {visualization_output['error']}", "history": history, "retries": state.get('retries', 0) + 1}
        
        # Update state with either visualization path or table data
        updated_state = {**state, "error": None, "history": history, "retries": 0}
        if 'visualization' in visualization_output:
            updated_state['visualization'] = visualization_output.get('visualization')
        if 'table' in visualization_output:
            updated_state['table'] = visualization_output.get('table')
            
        return updated_state

    except Exception as e:
        return {**state, "error": f"Error in Visualizer: {e}", "retries": state.get('retries', 0) + 1}

def summary_node(state: AgentState) -> AgentState:
    if state.get('error') and state.get('retries', 0) == 0: return state
    try:
        print("---GENERATING SUMMARY---")
        history = state.get('history', [])
        if state.get('error'):
            history.append({
                "error": state.get('error')
            })

        summary_agent = SummaryAgent(
            provider=state.get('llm_provider'),
            model_name=state.get('model_name'),
            google_api_key=state.get('google_api_key'),
            cohere_api_key=state.get('cohere_api_key')
        )
        summary_text = summary_agent.generate_summary(state['query'], state['execution_result'], state.get('dataset_hash'), state['sql_query'])
        return {**state, "summary": summary_text, "error": None, "history": history, "retries": 0}
    except Exception as e:
        return {**state, "error": f"Error in Summarizer: {e}", "retries": state.get('retries', 0) + 1}

def should_generate_visualization(state: AgentState) -> str:
    """Determines whether to generate a visualization or a summary."""
    if state.get('error'):
        return "end"
    
    print("---ROUTING---")
    chart_type = state.get('chart_type', 'none').lower()
    print(f"Chart type for routing: '{chart_type}'")
    
    if chart_type != 'none' and chart_type is not None and chart_type != 'table':
        print("Decision: Route to VISUALIZER")
        return "visualizer"
    else:
        print("Decision: Route to SUMMARIZER")
        return "summarizer"

def should_retry_visualization(state: AgentState) -> str:
    """Determines whether to retry visualization or continue."""
    print("---CHECKING FOR VISUALIZATION ERRORS---")
    retries = state.get('retries', 0)
    if state.get('error'):
        if retries < MAX_RETRIES:
            print(f"Visualization Error detected. Retry attempt {retries + 1}")
            # Increment retries and return to the same node
            state['retries'] = retries + 1
            return "retry"
        else:
            print("Visualization Error detected. Max retries reached. Aborting.")
            return "end"
    print("No visualization error. Proceeding.")
    # Reset retries on success and continue
    state['retries'] = 0
    return "continue"



def should_retry_summary(state: AgentState) -> str:
    """Determines whether to retry summary or end."""
    print("---CHECKING FOR SUMMARY ERRORS---")
    retries = state.get('retries', 0)
    if state.get('error'):
        if retries < MAX_RETRIES:
            print(f"Summary Error detected. Retry attempt {retries + 1}")
            # Increment retries and return to the same node
            state['retries'] = retries + 1
            return "retry"
        else:
            print("Summary Error detected. Max retries reached. Aborting.")
            return "end"
    print("No summary error. Proceeding.")
    # Reset retries on success and continue
    state['retries'] = 0
    return "continue"

def rejection_node(state: AgentState) -> AgentState:
    """Ends the process if the query is irrelevant."""
    print("---QUERY REJECTED---")
    return {**state, "summary": "I can only answer questions related to the uploaded data. Please ask a question about the available columns and data."}


def write_to_cache_node(state: AgentState) -> AgentState:
    """Saves the final response to both direct and semantic caches."""
    # Don't cache if the result came from a cache hit or if there was an error
    if state.get("direct_cache_hit") or state.get("semantic_cache_hit") or state.get("error"):
        return state

    print("---WRITING TO CACHE---")
    query = state.get('query')
    dataset_hash = state.get('dataset_hash')
    
    if not query or not dataset_hash:
        logger.warning("Could not write to cache, missing query or dataset_hash in state.")
        return state

    response_data = {}
    summary = state.get('summary')
    visualization = state.get('visualization')
    table = state.get('table')

    if summary is not None:
        response_data['summary'] = summary
    if visualization is not None:
        response_data['visualization'] = visualization
    if table is not None:
        response_data['table'] = table

    if response_data:
        # Direct cache write
        cache_key = f"query:{dataset_hash}:{query}"
        cache_manager.set(cache_key, response_data)
        logger.info(f"Saved to direct cache with key: {cache_key}")

        # Semantic cache write
        semantic_cache.add(query, response_data)
        logger.info(f"Saved to semantic cache for query: {query}")

    return state


def route_after_planner(state: AgentState) -> str:
    """Determines whether to retry the planner, or continue to SQL generation or rejection."""
    print("---CHECKING FOR PLANNER ERRORS AND ROUTING---")
    
    # Check for errors and decide on retries
    if state.get('error'):
        retries = state.get('retries', 0)
        if retries < MAX_RETRIES:
            print(f"Planner Error detected. Retry attempt {retries}")
            return "retry"
        else:
            print("Planner Error detected. Max retries reached. Aborting.")
            return "end"

    # If no errors, perform relevance check
    print("No planner error. Checking for relevance.")
    is_relevant = state.get("is_relevant", True) # Default to True if not specified
    if is_relevant:
        print("Decision: Query is relevant. Route to SQL_GENERATOR.")
        return "sql_generator"
    else:
        print("Decision: Query is irrelevant. Route to REJECTION.")
        return "rejection"

def get_graph_app():
    # Create the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("direct_cache", direct_cache_node)
    workflow.add_node("semantic_cache", semantic_cache_node)
    workflow.add_node("analysis_router", lambda state: state)
    workflow.add_node("schema_analyzer", schema_analysis_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("code_executor", code_executor_node)
    workflow.add_node("visualization_router", visualization_router_node)
    workflow.add_node("visualizer", visualization_node)
    workflow.add_node("summarizer", summary_node)
    workflow.add_node("rejection", rejection_node)
    workflow.add_node("write_to_cache", write_to_cache_node)

    # --- Define Edges ---

    # Entry point
    workflow.set_entry_point("direct_cache")

    # Cache routing
    workflow.add_conditional_edges(
        "direct_cache",
        route_after_direct_cache,
        {"semantic_cache": "semantic_cache", END: END}
    )
    workflow.add_conditional_edges(
        "semantic_cache",
        route_after_semantic_cache,
        {"analysis_router": "analysis_router", END: END}
    )

    # Conditional analysis routing
    workflow.add_conditional_edges(
        "analysis_router",
        route_to_analysis,
        {"schema_analyzer": "schema_analyzer", "planner": "planner"}
    )

    # Core pipeline flow
    workflow.add_conditional_edges(
        "schema_analyzer",
        should_retry_analysis,
        {
            "retry": "schema_analyzer",
            "continue": "planner",
            "end": END
        }
    )

    # Planner self-correction and relevance routing
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retry": "planner",
            "sql_generator": "sql_generator",
            "rejection": "rejection",
            "end": END
        }
    )
    
    workflow.add_edge("sql_generator", "code_executor")

    # SQL self-correction routing
    workflow.add_conditional_edges(
        "code_executor",
        should_retry_sql,
        {
            "retry": "sql_generator",
            "continue": "visualization_router",
            "end": END
        }
    )

    # Visualization vs. Summary routing
    workflow.add_conditional_edges(
        "visualization_router",
        should_generate_visualization,
        {
            "visualizer": "visualizer",
            "summarizer": "summarizer",
            "end": END
        }
    )

    # Visualizer self-correction routing
    workflow.add_conditional_edges(
        "visualizer",
        should_retry_visualization,
        {
            "retry": "visualizer",
            "continue": "summarizer",
            "end": END
        }
    )

    # Summarizer self-correction routing
    workflow.add_conditional_edges(
        "summarizer",
        should_retry_summary,
        {
            "retry": "summarizer",
            "continue": "write_to_cache",
            "end": END
        }
    )

    # Endpoints
    workflow.add_edge("rejection", END)
    workflow.add_edge("write_to_cache", END)

    # Compile the graph
    return workflow.compile()

graph_app = get_graph_app()
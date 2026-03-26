import matplotlib
matplotlib.use('Agg') # This line must be at the very top
import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.prompts import PromptTemplate
import base64
from io import BytesIO
import logging
import os
import uuid
import json
from config import Config
from utils.cache import cache
import seaborn as sns

logger = logging.getLogger(__name__)

class VisualizationAgent:
    def __init__(self, provider='google', model_name=None, google_api_key=None, cohere_api_key=None):
        logger.info("Initializing VisualizationAgent.")
        self.model_name = model_name
        self.llm = None
        
        # Initialize LLM for fallback axis selection
        try:
            self.llm = Config.get_llm(
                provider=provider, 
                model_name=model_name,
                google_api_key=google_api_key, 
                cohere_api_key=cohere_api_key
            )
        except Exception as e:
            logger.warning(f"Could not initialize LLM for Visualization fallback: {e}")

    def _get_axes_from_llm(self, df: pd.DataFrame, chart_type: str, history: list = None):
        """Fallback method: Asks the LLM to determine the best X and Y axes."""
        if not self.llm:
            logger.warning("No LLM available for visualization fallback.")
            return None, None

        logger.info("Using LLM fallback to determine chart axes.")
        columns_info = "\n".join([f"- {col} ({dtype})" for col, dtype in df.dtypes.items()])
        sample_df = df.head(3).copy()
        for col, dtype in sample_df.dtypes.items():
            if pd.api.types.is_datetime64_any_dtype(dtype):
                sample_df[col] = sample_df[col].dt.strftime('%Y-%m-%d')
        sample_data = sample_df.to_dict(orient='records')

        history_str = ""
        if history:
            for entry in history:
                history_str += f"Previous Error: {entry.get('error', '')}\n"

        prompt = PromptTemplate.from_template('''
        You are a data visualization assistant. Your ONLY task is to identify the best X and Y axes for a {chart_type} chart from the provided data.

        Data Schema:
        {columns_info}

        Data Sample:
        {sample_data}

        History of Previous Errors:
        {history}

        # RULES:
        1. The Y-axis MUST be a numeric column suitable for aggregation or measurement.
        2. The X-axis should be a categorical or temporal column for grouping or trending.
        3. Your response MUST be a single, valid JSON object.
        4. The JSON object MUST have two keys: "x_axis" and "y_axis".
        5. DO NOT include ANY text, explanation, or markdown formatting outside of the JSON object. Your entire response must be ONLY the JSON.

        Example of a perfect response:
        {{"x_axis": "Region", "y_axis": "Total_Sales"}}
        ''')

        try:
            formatted_prompt = prompt.format(
                chart_type=chart_type,
                columns_info=columns_info,
                sample_data=json.dumps(sample_data),
                history=history_str
            )
            response = self.llm.invoke(formatted_prompt)
            
            content = response if isinstance(response, str) else response.content
            content = content.strip()

            if not content:
                logger.error("LLM returned an empty response for axis selection.")
                return None, None

            # Attempt to find and parse a JSON object from the LLM's response
            try:
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start != -1 and json_end != 0:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    return result.get("x_axis"), result.get("y_axis")
                else:
                    raise json.JSONDecodeError("No JSON object found in response", content, 0)
            except json.JSONDecodeError:
                logger.error(f"LLM axis selection failed: Could not decode JSON from response: '{content}'.")
                return None, None
        except Exception as e:
            logger.error(f"An unexpected error occurred during LLM axis selection: {e}")
            return None, None

    def _create_scatter_plot(self, df, x_axis, y_axis):
        """Creates a scatter plot."""
        df.plot(kind='scatter', x=x_axis, y=y_axis, ax=plt.gca())

    def _create_pie_chart(self, df, x_axis, y_axis):
        """Helper to prepare data for a pie chart."""
        if df[x_axis].nunique() > 10: # Limit slices for readability
            plot_df = df.groupby(x_axis)[y_axis].sum().nlargest(10)
        else:
            plot_df = df.set_index(x_axis)[y_axis]
        plot_df.plot(kind='pie', autopct='%1.1f%%', ax=plt.gca(), legend=False)

    def _create_histogram(self, df, x_axis, y_axis=None): # y_axis is not used but kept for consistency
        """Creates a histogram."""
        df[x_axis].plot(kind='hist', ax=plt.gca(), bins=20)

    def _create_heatmap(self, df, x_axis, y_axis, value_col):
        """Creates a heatmap from the dataframe."""
        pivot_table = df.pivot(index=y_axis, columns=x_axis, values=value_col)
        sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="viridis", ax=plt.gca())

    def _select_axes(self, df: pd.DataFrame, chart_type: str) -> tuple:
        """Selects the best axes for a given chart type using a combination of heuristics and LLM."""
        x_axis, y_axis, value_col = None, None, None

        if chart_type in ['bar', 'line', 'scatter', 'pie']:
            x_axis, y_axis = self._get_axes_from_llm(df, chart_type)
        elif chart_type == 'histogram':
            # For a histogram, we only need one numeric axis.
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols:
                x_axis = numeric_cols[0]  # Simple heuristic is fine for histograms
        elif chart_type == 'heatmap':
            # Heatmaps are more complex and will rely on the LLM to find the best 3 columns
            # Note: The current _get_axes_from_llm only returns x and y. This would need to be extended for heatmaps.
            logger.warning("Heatmap axis selection currently relies on a simple heuristic. This may be improved in the future.")
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
            if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                x_axis = categorical_cols[0]
                y_axis = categorical_cols[1]
                value_col = numeric_cols[0]
        
        return x_axis, y_axis, value_col

    def generate_visualization(self, execution_result: dict, chart_type: str = 'bar', dataset_hash: str = '', query: str = '', history: list = None) -> dict:
        logger.info(f"Generating visualization with chart type: {chart_type}")

        # Check cache first
        cache_key = cache.get_chart_key(dataset_hash, query, chart_type)
        cached_chart_path = cache.get(cache_key)
        if cached_chart_path:
            logger.info(f"Cache hit for chart: {chart_type}")
            return {"visualization": cached_chart_path}
        
        if 'result' not in execution_result or not execution_result['result']:
            return {"error": "No result to visualize."}
        
        try:
            if isinstance(execution_result, dict) and 'result' in execution_result:
                df = pd.DataFrame(execution_result['result'])
            else:
                # If the input is not in the expected dict format, assume it's already a list of records
                df = pd.DataFrame(execution_result)
            if df.empty:
                return {"error": "Result is empty."}

            # If the result is a single value, override any chart request and present it as a table.
            if df.shape[0] == 1:
                logger.info("Execution result has only one row. Overriding chart type and presenting as a table.")
                table_data = df.to_dict(orient='split')
                # Do not cache this as a chart, as it's a fallback table representation
                return {"table": table_data}
            if chart_type == 'table':
                logger.info("Handling table generation. Returning structured data.")
                table_data = df.to_dict(orient='split')
                result = {"table": table_data}
                cache.set(cache_key, result)
                return result

            # --- Axis Selection Logic ---
            x_axis, y_axis, value_col = None, None, None

            # Always use the LLM to determine the best axes for the chart.
            if chart_type in ['bar', 'line', 'scatter', 'pie']:
                x_axis, y_axis = self._get_axes_from_llm(df, chart_type, history)
            elif chart_type == 'histogram':
                # For a histogram, we only need one numeric axis.
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    x_axis = numeric_cols[0] # Simple heuristic is fine for histograms
            elif chart_type == 'heatmap':
                # Heatmaps are more complex and will rely on the LLM to find the best 3 columns
                # Note: The current _get_axes_from_llm only returns x and y. This would need to be extended for heatmaps.
                logger.warning("Heatmap axis selection currently relies on a simple heuristic. This may be improved in the future.")
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
                if len(categorical_cols) >= 2 and len(numeric_cols) >= 1:
                    x_axis = categorical_cols[0]
                    y_axis = categorical_cols[1]
                    value_col = numeric_cols[0]

            # Validate the axes returned by the LLM
            if not x_axis or (chart_type not in ['histogram'] and not y_axis):
                logger.error(f"LLM failed to determine appropriate axes for chart type '{chart_type}'.")
                return {"error": f"Could not determine appropriate axes for the {chart_type} chart."}

            if x_axis not in df.columns or (y_axis and y_axis not in df.columns):
                logger.error(f"LLM returned invalid axes: X='{x_axis}', Y='{y_axis}'.")
                return {"error": "The AI model returned invalid column names for the chart axes."}

            plt.figure(figsize=(12, 7))

            chart_functions = {
                'bar': lambda: df.plot(kind='bar', x=x_axis, y=y_axis, ax=plt.gca()),
                'line': lambda: df.plot(kind='line', x=x_axis, y=y_axis, ax=plt.gca()),
                'scatter': lambda: self._create_scatter_plot(df, x_axis, y_axis),
                'pie': lambda: self._create_pie_chart(df, x_axis, y_axis),
                'histogram': lambda: self._create_histogram(df, x_axis),
                'heatmap': lambda: self._create_heatmap(df, x_axis, y_axis, value_col)
            }

            # Get the function from the dictionary and call it
            draw_chart = chart_functions.get(chart_type)
            if draw_chart:
                draw_chart()
            else:
                # Default to bar chart if type is unknown
                df.plot(kind='bar', x=x_axis, y=y_axis, ax=plt.gca())

            plt.title('Analysis Result')
            plt.xlabel(x_axis)
            if chart_type == 'pie':
                plt.ylabel('')
            elif chart_type == 'histogram':
                plt.ylabel('Frequency')
            else:
                plt.ylabel(y_axis)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save the plot to a file
            image_dir = os.path.join('static', 'images')
            os.makedirs(image_dir, exist_ok=True)
            image_filename = f"{uuid.uuid4()}.png"
            image_path = os.path.join(image_dir, image_filename)
            plt.savefig(image_path)
            plt.close()

            # Return the relative path for the web layer to handle
            relative_path = os.path.join('images', image_filename).replace('\\', '/')
            cache.set(cache_key, relative_path)
            return {"visualization": relative_path}
        except Exception as e:
            logger.error("Failed to generate plot", exc_info=True)
            return {"error": f"Failed to generate plot: {e}"}
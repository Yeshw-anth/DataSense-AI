from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    data_path: str
    dataset_hash: str
    table_name: str
    table_info: str
    example_rows: str
    history: List[Dict[str, Any]]
    plan: List[str]
    chart_type: str
    sql_query: str
    execution_result: dict
    visualization: str
    table: dict
    summary: str
    error: str
    llm_provider: str
    google_api_key: str
    cohere_api_key: str
    model_name: str
    is_relevant: bool
    semantic_cache_hit: bool
    retries: int
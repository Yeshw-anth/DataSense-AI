import duckdb
import pandas as pd
import logging
from utils.cache import cache

logger = logging.getLogger(__name__)

def execute_sql(sql_query: str, data_path: str, table_name: str, dataset_hash: str) -> dict:
    """
    Reads data from a file and executes a SQL query using DuckDB, with caching.

    Args:
        sql_query: The SQL query to execute.
        data_path: The path to the CSV or Excel file.
        table_name: The name to register the DataFrame as a virtual table.
        dataset_hash: The hash of the dataset file.

    Returns:
        A dictionary containing the result of the query or an error.
    """
    # Check cache first
    cache_key = cache.get_sql_key(dataset_hash, sql_query)
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for query: {sql_query}")
        return cached_result

    logger.info(f"Executing SQL query: {sql_query} on table '{table_name}' from file '{data_path}'")
    try:
        # 1. Prepare the data
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(data_path)
        else:
            return {"error": f"Unsupported file type: {data_path}"}
        
        # 2. Execute query
        conn = duckdb.connect(database=':memory:')
        conn.register(table_name, df)
        result_df = conn.execute(sql_query).fetchdf()
        
        # 3. Format and cache the result
        result = {"result": result_df.to_dict(orient='records')}
        cache.set(cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"DuckDB query failed: {e}", exc_info=True)
        return {"error": str(e)}
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cohere import ChatCohere
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from config import Config
from langchain_core.output_parsers import StrOutputParser
import re
import logging

logger = logging.getLogger(__name__)

class SQLAgent:
    def __init__(self, provider: str = 'google', model_name: str = None, google_api_key: str = None, cohere_api_key: str = None):
        logger.info(f"Initializing SQLAgent with provider: {provider}")
        self.llm = Config.get_llm(
            provider=provider, 
            model_name=model_name,
            google_api_key=google_api_key, 
            cohere_api_key=cohere_api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=['table_name', 'table_info', 'plan', 'history'],
            template=(
                "You are an expert DuckDB data analyst. Your only purpose is to write a single, syntactically correct DuckDB query that accomplishes the goal of the provided plan."
                "\n# Instructions:"
                "\n1. **Your ONLY task is to translate the following high-level plan into a single, valid DuckDB query.**"
                "\n2. **Use the provided table context to inform the query.**"
                "\n3. **The query MUST be a single statement.**"
                "\n4. **The query must be directly executable on a DuckDB database.**"
                "\n5. **Analyze the user's query history to understand the context and avoid repeating previous mistakes.**"

                "\n\n# DuckDB Best Practices:"
                "\n- **Date Filtering:** When filtering by year, do NOT use `STRPTIME`. This function returns a `TIMESTAMP`, which will cause a type error when compared to an integer year. Instead, cast the date column to `DATE` and use the `YEAR()` function."
                "\n  - **INCORRECT:** `... WHERE STRPTIME(\"Date\", '%Y') BETWEEN 1995 AND 2005`"
                "\n  - **CORRECT:** `... WHERE YEAR(CAST(\"Date\" AS DATE)) BETWEEN 1995 AND 2005`"

                "\n- **Aggregating Window Functions:** You CANNOT nest a window function (like `LAG()` or `LEAD()`) inside an aggregate function (like `MAX()` or `SUM()`). To accomplish this, you MUST use a Common Table Expression (CTE)."
                "\n  - **INCORRECT:** `SELECT MAX(col - LAG(col) OVER(...)) FROM table`"
                "\n  - **CORRECT:** `WITH Changes AS (SELECT col - LAG(col) OVER(...) as change FROM table) SELECT MAX(change) FROM Changes;`"

                "\n- **Combining GROUP BY with Window Functions:** You CANNOT use a window function (like `FIRST_VALUE` or `LAG`) in a query that also has a `GROUP BY` clause unless the column from the window function is also in the `GROUP BY`. The correct way to handle this is to calculate the window function in a CTE first, and then perform the `GROUP BY` in the outer query."
                "\n  - **INCORRECT:** `SELECT YEAR(Date), FIRST_VALUE(Price) OVER (ORDER BY Date) FROM table GROUP BY YEAR(Date)`"
                "\n  - **CORRECT:** `WITH InitialPrices AS (SELECT YEAR(Date) as year, Price, ROW_NUMBER() OVER (PARTITION BY YEAR(Date) ORDER BY Date) as rn FROM table) SELECT year, Price FROM InitialPrices WHERE rn = 1;`"

                "\n- **Calculating Period-Over-Period Change:** To get the first and last values of a column over a period to calculate change, do NOT mix aggregates like `MIN()`/`MAX()` with window functions like `FIRST_VALUE()`. Instead, use `ROW_NUMBER()` in a CTE to rank rows by date and then select the first and last ones."
                "\n  - **INCORRECT:** `SELECT MIN(Date), FIRST_VALUE(Price) OVER (ORDER BY Date) FROM table`"
                "\n  - **CORRECT:** `WITH RankedPrices AS (SELECT Price, ROW_NUMBER() OVER (ORDER BY Date ASC) as rn_asc, ROW_NUMBER() OVER (ORDER BY Date DESC) as rn_desc FROM table) SELECT MAX(CASE WHEN rn_asc = 1 THEN Price END) as FirstPrice, MAX(CASE WHEN rn_desc = 1 THEN Price END) as LastPrice FROM RankedPrices;`"

                "\n\n# Context:"
                "\nYou will be querying a table with the following details:"
                "\n- **Table Name:** `{table_name}`"
                "\n- **Table Info (Columns and Types):**\n{table_info}"
                "\n- **Execution Plan:**\n{plan}\n"
                "\n- **Query History:**\n{history}"
                "\n# DuckDB Query:"
            )
        )
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def _extract_sql(self, text: str) -> str:
        """Extracts the SQL query from a markdown code block, supporting both 'sql' and 'duckdb'."""
        # The pattern now looks for ```sql, ```duckdb, or just ``` followed by the query.
        match = re.search(r"```(sql|duckdb)?\n(.*?)\n```", text, re.DOTALL)
        if match:
            # The actual query is in the second capturing group.
            return match.group(2).strip()
        # Fallback for cases where the model doesn't use markdown
        return text.strip()

    def generate_sql(self, table_name: str, table_info: str, plan: list, history: list = None) -> str:
        """Generates a SQL query to answer a given query using LangChain and Ollama."""
        history_str = ""
        if history:
            for entry in history:
                history_str += f"Previous Query: {entry.get('sql_query', '')}\nError: {entry.get('error', '')}\n"

        response = self.chain.invoke({
            'table_name': table_name, 
            'table_info': table_info, 
            'plan': "\n".join(f"- {step}" for step in plan),
            'history': history_str
        })
        sql_query = self._extract_sql(response)
        logger.info(f"LLM-generated SQL: {sql_query}")
        return sql_query
from langchain_core.prompts import PromptTemplate
from config import Config
from langchain_core.output_parsers import StrOutputParser
import logging
from utils.cache import cache

logger = logging.getLogger(__name__)

class SummaryAgent:
    def __init__(self, provider='google', model_name=None, google_api_key=None, cohere_api_key=None):
        logger.info(f"Initializing SummaryAgent with provider: {provider}")
        self.llm = Config.get_llm(
            provider=provider, 
            model_name=model_name,
            google_api_key=google_api_key, 
            cohere_api_key=cohere_api_key
        )
        self.prompt = PromptTemplate(
            input_variables=['query', 'results'],
            template=(
                "You are an expert data analyst. Your job is to provide a clear, natural-language summary of the results of a SQL query. "
                "The user's query was: '{query}'"
                "The results of the query are: \n{results}"
                "\n---\n"
                "Based on the query and the results, provide a concise, easy-to-understand summary."
                "\nSummary:"
            )
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_summary(self, query: str, results: dict, dataset_hash: str, sql_query: str, summary_version: str = 'v1') -> str:
        if not results or 'result' not in results or not results['result']:
            logger.warning("No results to summarize.")
            return "No results to summarize."

        # Check cache first
        cache_key = cache.get_summary_key(dataset_hash, query, summary_version)
        cached_summary = cache.get(cache_key)
        if cached_summary:
            logger.info(f"Cache hit for summary: {query}")
            return cached_summary

        results_str = "\n".join([str(row) for row in results['result']])

        response = self.chain.invoke({"query": query, "results": results_str})

        # Cache the result
        cache.set(cache_key, response)

        return response
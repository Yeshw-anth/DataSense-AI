import pytest
from agents.summary_agent import SummaryAgent
import logging

# Configure logging to avoid "No handler found" warnings during testing
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def summary_agent():
    """Provides a SummaryAgent instance for testing."""
    # We can use the default 'google' provider since we will mock the LLM call
    return SummaryAgent(model_name='google')

def test_generate_summary_success(summary_agent, mocker):
    """
    Tests that the SummaryAgent can successfully generate a summary.
    """
    # Arrange
    query = "What are the top 5 products by sales?"
    results = {
        "result": [
            {'product': 'A', 'sales': 100},
            {'product': 'B', 'sales': 90},
            {'product': 'C', 'sales': 80},
            {'product': 'D', 'sales': 70},
            {'product': 'E', 'sales': 60},
        ]
    }
    expected_summary = "The top 5 products by sales are A, B, C, D, and E."

    # Mock the LangChain RunnableSequence's invoke method to return a predictable response
    mock_invoke = mocker.patch(
        'langchain_core.runnables.base.RunnableSequence.invoke', 
        return_value=expected_summary
    )

    # Act
    summary = summary_agent.generate_summary(query, results, dataset_hash='dummy_hash', sql_query='dummy_sql')

    # Assert
    assert summary == expected_summary
    mock_invoke.assert_called_once()

def test_generate_summary_no_results(summary_agent):
    """
    Tests that the SummaryAgent returns a specific message when there are no results.
    """
    # Arrange
    query = "What are the sales for a non-existent product?"
    empty_results = {"result": []}
    
    # Act
    summary = summary_agent.generate_summary(query, empty_results, dataset_hash='dummy_hash', sql_query='dummy_sql')

    # Assert
    assert summary == "No results to summarize."

def test_generate_summary_malformed_results(summary_agent):
    """
    Tests that the SummaryAgent returns a specific message for malformed results.
    """
    # Arrange
    query = "Any query"
    malformed_results = {} # Missing the 'result' key
    
    # Act
    summary = summary_agent.generate_summary(query, malformed_results, dataset_hash='malformed_hash', sql_query='malformed_query')

    # Assert
    assert summary == "No results to summarize."
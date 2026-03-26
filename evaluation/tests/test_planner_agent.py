
import pytest
from agents.planner_agent import PlannerAgent
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def planner_agent():
    """Provides a PlannerAgent instance for testing."""
    return PlannerAgent(model_name='google')

def test_create_plan_success(planner_agent, mocker):
    """
    Tests that the PlannerAgent can successfully create a plan and suggest a chart type.
    """
    # Arrange
    query = "What are the top 10 countries by population?"
    history = []
    table_name = "countries"
    table_info = "- country (text)\n- population (integer)"
    example_rows = "country,population\nUSA,330\nChina,1400"

    expected_response_json = {
        "is_relevant": True,
        "plan": [
            "Filter the data to include only the top 10 countries by population",
            "Order the results in descending order based on population"
        ],
        "chart_type": "bar"
    }
    # The LLM often returns JSON as a string inside a markdown block
    mock_llm_output = f"```json\n{json.dumps(expected_response_json)}\n```"

    # Mock the LangChain RunnableSequence's invoke method
    mock_invoke = mocker.patch(
        'langchain_core.runnables.base.RunnableSequence.invoke', 
        return_value=mock_llm_output
    )

    # Act
    result = planner_agent.create_plan(query, history, table_name, table_info, example_rows)

    # Assert
    assert result == expected_response_json
    mock_invoke.assert_called_once()

def test_create_plan_json_parsing_fallback(planner_agent, mocker):
    """
    Tests that the agent correctly handles a malformed JSON response from the LLM
    and falls back to default values.
    """
    # Arrange
    query = "Any query"
    mock_llm_output = "This is not JSON, just a plain string."

    mocker.patch(
        'langchain_core.runnables.base.RunnableSequence.invoke', 
        return_value=mock_llm_output
    )

    # Act
    result = planner_agent.create_plan(query, [], "", "", "")

    # Assert
    # The agent should return a default structure when parsing fails
    assert result['plan'] == []
    assert result['chart_type'] == 'table'

import pytest
from agents.sql_agent import SQLAgent
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sql_agent():
    """Provides a SQLAgent instance for testing."""
    return SQLAgent(model_name='google')

def test_generate_sql_success(sql_agent, mocker):
    """
    Tests that the SQLAgent can successfully generate a SQL query from a plan.
    """
    # Arrange
    table_name = "sales"
    table_info = "- product (text)\n- amount (integer)"
    example_rows = "product,amount\nLaptop,1200\nMouse,25"
    plan = [
        "Calculate the total sales for each product",
        "Order the results by the total sales in descending order"
    ]
    
    mock_llm_output = "```sql\nSELECT product, SUM(amount) AS total_sales FROM sales GROUP BY product ORDER BY total_sales DESC;\n```"
    expected_sql = "SELECT product, SUM(amount) AS total_sales FROM sales GROUP BY product ORDER BY total_sales DESC;"

    # Mock the LangChain RunnableSequence's invoke method
    mock_invoke = mocker.patch(
        'langchain_core.runnables.base.RunnableSequence.invoke',
        return_value=mock_llm_output
    )

    # Act
    result_sql = sql_agent.generate_sql(table_name, table_info, example_rows, plan)

    # Assert
    assert result_sql == expected_sql
    mock_invoke.assert_called_once()

def test_sql_extraction_no_markdown(sql_agent, mocker):
    """
    Tests that the SQL extraction works even if the LLM omits the markdown block.
    """
    # Arrange
    plan = ["Find all products"] # Plan doesn't matter for this test
    mock_llm_output = "SELECT * FROM sales;"
    expected_sql = "SELECT * FROM sales;"

    mocker.patch(
        'langchain_core.runnables.base.RunnableSequence.invoke',
        return_value=mock_llm_output
    )

    # Act
    result_sql = sql_agent.generate_sql("sales", "", "", plan)

    # Assert
    assert result_sql == expected_sql
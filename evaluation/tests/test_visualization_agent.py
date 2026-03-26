import pytest
import pandas as pd
import os
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from agents.visualization_agent import VisualizationAgent

@pytest.fixture
def visualization_agent():
    """Provides an instance of the VisualizationAgent for testing."""
    return VisualizationAgent()

@pytest.fixture
def sample_execution_result():
    """Provides a sample execution result dictionary."""
    data = {
        'product_category': ['Electronics', 'Clothing', 'Books', 'Home Goods'],
        'sales': [1500, 1200, 800, 2000]
    }
    return {"result": pd.DataFrame(data).to_dict(orient='records')}

def test_generate_table(visualization_agent, sample_execution_result):
    """
    Tests if the agent correctly returns structured data for a 'table' chart type.
    """
    chart_type = 'table'
    result = visualization_agent.generate_visualization(sample_execution_result, chart_type)

    # 1. Check that the output is a dictionary containing the 'table' key
    assert isinstance(result, dict)
    assert 'table' in result
    assert 'visualization' not in result # Ensure no image is generated

    # 2. Check the structure of the table data
    table_data = result['table']
    assert 'columns' in table_data
    assert 'data' in table_data
    assert table_data['columns'] == ['product_category', 'sales']
    assert len(table_data['data']) == 4

def test_generate_bar_chart(visualization_agent, sample_execution_result):
    """
    Tests if the agent correctly generates a bar chart and returns a valid file path.
    """
    chart_type = 'bar'
    result = visualization_agent.generate_visualization(sample_execution_result, chart_type)

    # 1. Check that the output is a dictionary containing the 'visualization' key
    assert isinstance(result, dict)
    assert 'visualization' in result
    assert 'table' not in result # Ensure no table data is generated

    # 2. Check that the returned path is a valid file
    image_path = result['visualization']
    expected_path_start = '/static/images/'
    assert isinstance(image_path, str)
    assert image_path.startswith(expected_path_start)
    assert image_path.endswith('.png')

    # Convert URL path to system-native path for existence check and cleanup
    # e.g., /static/images/foo.png -> static\images\foo.png on Windows
    file_path = os.path.join(*image_path.strip("/").split("/"))
    assert os.path.exists(file_path)

    # Clean up the generated file
    os.remove(file_path)

def test_generate_scatter_chart(visualization_agent, sample_execution_result):
    """
    Tests if the agent correctly generates a scatter chart.
    """
    chart_type = 'scatter'
    result = visualization_agent.generate_visualization(sample_execution_result, chart_type)
    assert 'visualization' in result
    file_path = os.path.join(*result['visualization'].strip("/").split("/"))
    assert os.path.exists(file_path)
    os.remove(file_path)

def test_generate_pie_chart(visualization_agent, sample_execution_result):
    """
    Tests if the agent correctly generates a pie chart.
    """
    chart_type = 'pie'
    result = visualization_agent.generate_visualization(sample_execution_result, chart_type)
    assert 'visualization' in result
    file_path = os.path.join(*result['visualization'].strip("/").split("/"))
    assert os.path.exists(file_path)
    os.remove(file_path)

def test_generate_histogram_chart(visualization_agent, sample_execution_result):
    """
    Tests if the agent correctly generates a histogram.
    """
    chart_type = 'histogram'
    result = visualization_agent.generate_visualization(sample_execution_result, chart_type)

    assert 'visualization' in result
    file_path = os.path.join(*result['visualization'].strip("/").split("/"))
    assert os.path.exists(file_path)
    os.remove(file_path)

@pytest.fixture
def sample_heatmap_execution_result():
    """Provides a sample execution result suitable for a heatmap."""
    data = {
        'Month': ['Jan', 'Feb', 'Jan', 'Feb'],
        'City': ['CityA', 'CityA', 'CityB', 'CityB'],
        'Temperature': [10, 12, 15, 18]
    }
    return {"result": pd.DataFrame(data).to_dict(orient='records')}


def test_generate_heatmap_chart(visualization_agent, sample_heatmap_execution_result):
    """
    Tests if the agent correctly generates a heatmap.
    """
    chart_type = 'heatmap'
    result = visualization_agent.generate_visualization(sample_heatmap_execution_result, chart_type)

    assert 'visualization' in result
    file_path = os.path.join(*result['visualization'].strip("/").split("/"))
    assert os.path.exists(file_path)
    os.remove(file_path)
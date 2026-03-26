
import pytest
import json
from unittest.mock import patch
import pandas as pd
import os

from app import create_app
from extensions import cache

# Define a sample CSV for testing
SAMPLE_CSV_DATA = "product,category,sales\nLaptop,Electronics,1500\nMouse,Electronics,25\nKeyboard,Electronics,75\nDesk,Furniture,300\nChair,Furniture,150"

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['CACHE_TYPE'] = 'SimpleCache'
    app.cache = cache
    app.cache.init_app(app)
    with app.test_client() as client:
        with app.app_context():
            cache.clear()
        yield client

@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(SAMPLE_CSV_DATA)
    return str(csv_path)

@patch('pipelines.main_pipeline.graph_app.invoke')
@patch('utils.cache.semantic_cache.add')
@patch('utils.cache.semantic_cache.search')
def test_caching_layers(mock_semantic_search, mock_semantic_add, mock_invoke, client, temp_csv_file):
    """
    Tests the functionality of both repetitive and semantic caching layers using a real temp file.
    """
    # Mock setup
    mock_invoke.return_value = {
        "summary": "This is a test response.",
        "semantic_cache_hit": False
    }
    # No semantic match initially
    mock_semantic_search.return_value = ([], [])

    query_data = {
        "query": "What is the total revenue?",
        "filepath": temp_csv_file,
        "table_name": "sales"
    }

    # 1. First request: Cache miss for both layers
    print("\n--- Testing: First Request (Cache Miss) ---")
    response1 = client.post('/api/query', data=json.dumps(query_data), content_type='application/json')
    assert response1.status_code == 200
    mock_invoke.assert_called_once()
    mock_semantic_add.assert_called_once()
    print("--- Result: Cache Miss, pipeline invoked. Correct. ---")

    # 2. Second request: Repetitive cache hit
    print("\n--- Testing: Second Request (Repetitive Cache Hit) ---")
    response2 = client.post('/api/query', data=json.dumps(query_data), content_type='application/json')
    assert response2.status_code == 200
    # Assert that the pipeline was NOT called again
    mock_invoke.assert_called_once() # Still called only once in total
    print("--- Result: Repetitive Cache Hit, pipeline skipped. Correct. ---")

    # 3. Third request: Semantic cache hit
    print("\n--- Testing: Third Request (Semantic Cache Hit) ---")
    # Simulate a semantic match for a similar query
    similar_query_data = {
        "query": "Show me the total revenue.",
        "filepath": temp_csv_file,
        "table_name": "sales"
    }
    # Now, semantic search finds a match
    mock_semantic_search.return_value = ([{'response': {'summary': 'This is a test response.'}}], [0.95])
    
    # Clear the repetitive cache to ensure we are testing the semantic layer
    with client.application.app_context():
        cache.clear()

    response3 = client.post('/api/query', data=json.dumps(similar_query_data), content_type='application/json')
    assert response3.status_code == 200
    # The pipeline should not be invoked due to the semantic hit
    mock_invoke.assert_called_once() # Still only called once
    print("--- Result: Semantic Cache Hit, pipeline skipped. Correct. ---")
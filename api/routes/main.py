from flask import Blueprint, jsonify, request, render_template, url_for
from flask import current_app
from extensions import cache
import os
import pandas as pd
import re
import logging
from config import setup_logging
from data.processing import save_uploaded_file, get_dataset_hash
from google.api_core.exceptions import PermissionDenied as GoogleAuthenticationError
from cohere.errors import UnauthorizedError as CohereAuthenticationError


# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)

def _clean_column_names(df):
    """Cleans column names to be valid identifiers."""
    new_columns = {}
    for col in df.columns:
        new_col = re.sub(r'\W|^(?=\d)', '_', col)
        new_columns[col] = new_col
    df.rename(columns=new_columns, inplace=True)
    return df

@main_bp.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@main_bp.route('/api/health')
def health_check():
    """Provides a simple health check endpoint."""
    return jsonify({"status": "ok"}), 200


@main_bp.route('/api/upload', methods=['POST'])
def upload_file():
    """Handles file uploads, preparing the data for analysis."""
    # --- Delete previous file if it exists ---
    previous_filepath = request.form.get('previous_filepath')
    if previous_filepath and os.path.exists(previous_filepath):
        try:
            os.remove(previous_filepath)
            logger.info(f"Successfully deleted previous file: {previous_filepath}")
        except Exception as e:
            logger.warning(f"Could not delete previous file {previous_filepath}: {e}")

    # --- Clear the images directory ---
    images_dir = os.path.join('static', 'images')
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            file_path = os.path.join(images_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Use the new data processing module to save the file
        filepath = save_uploaded_file(file)

        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        df = _clean_column_names(df)
        # Overwrite the file with cleaned columns to ensure consistency
        df.to_csv(filepath, index=False)

        table_name = re.sub(r'\W|^(?=\d)', '_', os.path.splitext(os.path.basename(filepath))[0])

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'filepath': filepath,
            'table_name': table_name,
            'columns': list(df.columns)
        })
    except Exception as e:
        logger.error(f"Error during file upload: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@main_bp.route('/api/query', methods=['POST'])
def handle_query():
    from pipelines.main_pipeline import graph_app
    from utils.cache import semantic_cache
    """Handles a conversational query about the previously uploaded data."""
    data = request.get_json()
    query = data.get('query')
    filepath = data.get('filepath')
    table_name = data.get('table_name')
    history = data.get('history', [])
    llm_provider = data.get('llm_provider', 'google')
    google_api_key = data.get('google_api_key')
    cohere_api_key = data.get('cohere_api_key')
    model_name = data.get('model_name') # New

    if not all([query, filepath, table_name]):
        return jsonify({'error': 'Missing required fields'}), 400

    # --- API Key Validation ---
    if llm_provider == 'google' and not google_api_key:
        return jsonify({'error': 'Google API key is missing. Please provide it to proceed.'}), 400
    if llm_provider == 'cohere' and not cohere_api_key:
        return jsonify({'error': 'Cohere API key is missing. Please provide it to proceed.'}), 400



    try:
        dataset_hash = get_dataset_hash(filepath)
        inputs = {
            "query": query,
            "data_path": filepath, # Pass the file path instead of raw data
            "dataset_hash": dataset_hash, # Add dataset hash to the state
            "history": history,
            "table_name": table_name,
            "llm_provider": llm_provider,
            "model_name": model_name, # New
            "google_api_key": google_api_key,
            "cohere_api_key": cohere_api_key
        }

        logger.info(f"Invoking graph_app for table '{table_name}' with provider '{llm_provider}'.")
        final_state = graph_app.invoke(inputs)

        if final_state.get("error"):
            raise Exception(final_state["error"])

        response_data = {}
        summary = final_state.get('summary')
        visualization = final_state.get('visualization')
        table = final_state.get('table')

        if summary:
            response_data['summary'] = summary

        if visualization:
            # The agent returns a relative path, e.g., 'images/chart.png'
            # We use url_for to create the correct public URL
            response_data['visualization'] = url_for('static', filename=visualization)
            if not summary:
                response_data['summary'] = "Here is the generated visualization."
        
        if table:
            response_data['table'] = table

        return jsonify(response_data)

    except (GoogleAuthenticationError, CohereAuthenticationError) as e:
        logger.error(f"Authentication error during agent execution: {e}", exc_info=True)
        return jsonify({'error': 'API Key is invalid or has expired. Please check your key and try again.'}), 401
    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)
        return jsonify({'error': 'An internal error occurred on the server. The technical details have been logged for the administrator.'}), 500
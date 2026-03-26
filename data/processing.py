
import os
from werkzeug.utils import secure_filename
import hashlib

# Define the base directory for storing uploaded files
STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'storage')

def save_uploaded_file(file) -> str:
    """
    Saves an uploaded file to the storage directory and returns its path.

    Args:
        file: The file object from the request.

    Returns:
        The absolute path to the saved file.
    """
    if not file or not file.filename:
        raise ValueError("Invalid file provided.")

    # Ensure the filename is secure
    filename = secure_filename(file.filename)
    
    # Ensure the storage directory exists
    os.makedirs(STORAGE_DIR, exist_ok=True)
    
    # Create the full path and save the file
    file_path = os.path.join(STORAGE_DIR, filename)
    file.save(file_path)
    
    return file_path

def get_dataset_hash(filepath: str) -> str:
    """Computes the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
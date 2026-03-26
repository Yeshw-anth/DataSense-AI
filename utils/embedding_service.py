from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    """
    Lazily loads and returns the sentence-transformer model.

    This function ensures the model is loaded only once. On the first call,
    it loads the model and stores it in the global '_embedding_model' variable.
    All subsequent calls will return the already-loaded model instantly from memory.
    """
    global _embedding_model

    if _embedding_model is None:
        logger.info("Embedding model not found in memory. Loading 'all-MiniLM-L6-v2' for the first time...")
        try:
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully and cached in memory.")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformer model: {e}", exc_info=True)
            raise

    return _embedding_model
import logging
import numpy as np
import faiss
from .embedding_service import get_embedding_model
from collections import deque
import os
from abc import ABC, abstractmethod
import hashlib

def get_dataset_hash(filepath: str) -> str:
    """Generates a SHA256 hash for a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return "file_not_found"

logger = logging.getLogger(__name__)

SEMANTIC_CACHE_ENABLED = os.getenv('SEMANTIC_CACHE_ENABLED', 'True').lower() in ('true', '1', 't')
MAX_ENTRIES = int(os.getenv('SEMANTIC_CACHE_MAX_ENTRIES', 10))
SIMILARITY_THRESHOLD = float(os.getenv('SEMANTIC_CACHE_SIMILARITY_THRESHOLD', 0.85))

class BaseSemanticCache(ABC):
    @abstractmethod
    def add(self, query: str, response: dict):
        pass

    @abstractmethod
    def search(self, query: str):
        pass

class DisabledSemanticCache(BaseSemanticCache):
    def __init__(self):
        logger.info("Semantic cache is DISABLED.")

    def add(self, query: str, response: dict):
        pass

    def search(self, query: str):
        return None

class InMemorySemanticCache(BaseSemanticCache):
    def __init__(self, max_entries=100, similarity_threshold=SIMILARITY_THRESHOLD, embedding_dim=384):
        logger.info(f"Initializing InMemorySemanticCache with max_entries={max_entries}, similarity_threshold={similarity_threshold}")
        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.entries = deque(maxlen=max_entries)
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def _rebuild_index(self):
        self.index.reset()
        if self.entries:
            embeddings = np.array([entry["embedding"] for entry in self.entries]).astype('float32')
            self.index.add(embeddings)

    def add(self, query: str, response: dict):
        model = get_embedding_model()
        embedding = model.encode([query])[0]

        if len(self.entries) == self.max_entries:
            self.entries.popleft()

        self.entries.append({"query": query, "response": response, "embedding": embedding})
        self._rebuild_index()
        logger.info(f"Added to in-memory cache: '{query}'")

    def search(self, query: str):
        if not self.entries:
            return None

        model = get_embedding_model()
        query_embedding = model.encode([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), 1)

        if len(indices) > 0 and len(indices[0]) > 0:
            distance = distances[0][0]
            if distance >= 0:
                similarity = 1 / (1 + distance)

                if similarity > self.similarity_threshold:
                    cached_entry = self.entries[indices[0][0]]
                    logger.info(f"In-memory cache hit for query: '{query}' with similarity {similarity:.4f}.")
                    return cached_entry['response']

        return None

def create_semantic_cache():
    if not SEMANTIC_CACHE_ENABLED:
        return DisabledSemanticCache()

    return InMemorySemanticCache(
        max_entries=MAX_ENTRIES,
        similarity_threshold=SIMILARITY_THRESHOLD
    )

semantic_cache = create_semantic_cache()

class CacheManager:
    def get_sql_key(self, dataset_hash: str, sql_query: str) -> str:
        return f"sql:{dataset_hash}:{sql_query}"

    def get_summary_key(self, dataset_hash: str, query: str, summary_version: str) -> str:
        return f"summary:{dataset_hash}:{query}:{summary_version}"

    def get_chart_key(self, dataset_hash: str, query: str, chart_type: str) -> str:
        return f"chart:{dataset_hash}:{query}:{chart_type}"

    def get(self, key: str):
        from extensions import cache
        return cache.get(key)

    def set(self, key: str, value):
        from extensions import cache
        cache.set(key, value)

cache = CacheManager()
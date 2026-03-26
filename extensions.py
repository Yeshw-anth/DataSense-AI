import os
from flask_caching import Cache

# Default to filesystem cache if not specified
CACHE_TYPE = os.environ.get("CACHE_TYPE", "filesystem")

config = {
    "CACHE_TYPE": CACHE_TYPE,
    "CACHE_DEFAULT_TIMEOUT": 300
}

if CACHE_TYPE == "redis":
    config["CACHE_REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6379")
    # Use brotli compression for better performance with Redis
    config["CACHE_COMPRESS"] = True
    config["CACHE_COMPRESSION_LEVEL"] = 6 # Brotli compression level
elif CACHE_TYPE == "filesystem":
    config["CACHE_DIR"] = "cache"

cache = Cache(config=config)
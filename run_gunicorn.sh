#!/bin/sh
echo "Starting server with Gunicorn..."

# Use --preload to load app before worker fork, but embedding model is lazy-loaded in embedding_service.py
gunicorn --workers 4 \
         --bind 0.0.0.0:${PORT:-8000} \
         --preload \
         --log-level info \
         "app:create_app()"
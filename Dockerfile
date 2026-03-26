FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache embeddings to speed up first request
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN chmod +x run_gunicorn.sh

EXPOSE 8000

CMD ["./run_gunicorn.sh"]
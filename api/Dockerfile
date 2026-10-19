FROM python:3.12-slim

WORKDIR /app

# Install system library required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir --upgrade pip
RUN python -m pip install --no-cache-dir --retries 5 --timeout 120 -r requirements-api.txt

COPY api/ ./api/
COPY model.pkl ./model.pkl

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

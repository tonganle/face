FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FACE_ENGINE=onnx \
    FACE_DATA_DIR=/app/data

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app ./app
COPY scripts ./scripts
COPY datasets ./datasets
COPY README.md README_CN.md ./

EXPOSE 8000

CMD ["sh", "-c", "python scripts/download_onnx_models.py && python scripts/download_detector_model.py && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"]

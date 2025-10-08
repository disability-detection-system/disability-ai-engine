FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for OpenCV runtime and any optional audio libs used by the repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libgl1 gcc g++ make && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["gunicorn","--chdir", "api", "--bind", "0.0.0.0:5001", "handwriting_api:app"]

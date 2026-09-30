FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy backend application
COPY backend/ .

# The model, models and templates folders are already inside backend
# so they are included by the COPY above.

EXPOSE 5000

CMD ["python", "app.py"]
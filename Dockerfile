# Use official lightweight Python 3.10 base image
FROM python:3.10-slim

# Set environment variables for clean output and TensorFlow runtime suppression
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0

# Set working directory inside the container
WORKDIR /app

# Install system utilities needed for C++ native image decoding libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to maximize Docker layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies (NumPy < 2.0, TensorFlow, Gunicorn)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Flask application code and saved model binary into the container
COPY app.py .
COPY densenet_tb_model.keras .

# Expose Port 5001 (avoids macOS AirPlay on 5000 and Express.js on 3000)
EXPOSE 5001

# Run production Gunicorn server:
# - 1 worker + 2 threads: Prevents thread contention & duplicating model weights in RAM
# - 120s timeout: Allows ample time for model loading and initial graph warmup
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Cloud Run sets $PORT automatically (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]

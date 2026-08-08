FROM python:3.11-slim

WORKDIR /app

# System dependencies needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-build Matplotlib's font cache now, during the image build (fast, unthrottled),
# so it never has to rebuild it at container startup, where CPU may be limited
RUN python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure(); plt.savefig('/tmp/warmup.png')"

COPY app.py .

# Cloud Run sets $PORT automatically (defaults to 8080)
ENV PORT=8080
# Skip Matplotlib's slow font-cache build on every cold start — pulled in
# indirectly by ultralytics, not used directly by this app
ENV MPLBACKEND=Agg
EXPOSE 8080

CMD ["python", "app.py"]

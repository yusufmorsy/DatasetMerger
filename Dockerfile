FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by PyTorch (Debian-based)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
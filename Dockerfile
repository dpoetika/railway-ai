FROM python:3.13.0

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bağımlılıkları yükle
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY . .

# Hugging Face Spaces portu
EXPOSE 7860

# Flask başlat
CMD ["python", "app.py"]

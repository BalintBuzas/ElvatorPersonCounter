FROM python:3.10-slim

# Cache busting argumentum
ARG CACHEBUST=1

# Környezeti változók beállítása
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Munkakönyvtár beállítása
WORKDIR /app

# Alapvető rendszercsomagok telepítése
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Requirements telepítése (cache busting alkalmazása)
COPY requirements.txt .
RUN echo "$CACHEBUST" > /dev/null && pip install --no-cache-dir -r requirements.txt

# Projekt fájlok másolása
COPY . .

# Könyvtárak létrehozása a súlyok és a kimenetek számára
RUN mkdir -p weights runs

# Jogosultságok biztosítása a script számára
RUN chmod +x main.py

# Entrypoint és parancs beállítása
ENTRYPOINT ["python", "main.py"]
CMD ["--weights", "weights/crowdhuman.onnx", "--source", "assets/input.mp4"]

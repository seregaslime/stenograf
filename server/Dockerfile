# Сервер Стенографа: FastAPI + faster-whisper + silero-vad + ECAPA.
# Сценарий: сервер организации (Linux), клиенты подключаются по сети.
FROM python:3.12-slim

WORKDIR /srv

# Слой зависимостей кешируется отдельно от кода.
# torch ставим CPU-сборку с индекса PyTorch — без CUDA образ в разы меньше.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app

# Все данные (БД, модели, образцы голосов) — в томе /data
ENV STENOGRAF_DATA_DIR=/data
VOLUME /data

EXPOSE 8765
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]

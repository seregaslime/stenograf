# Сервер Стенографа: FastAPI + GigaAM/whisper + silero-vad + ECAPA.
# Сценарий: сервер организации (Linux), клиенты подключаются по сети.
# Сборка/запуск — из корня репозитория: `docker compose up -d --build`.
FROM python:3.12-slim

WORKDIR /srv

# Слой зависимостей кешируется отдельно от кода.
# torch — CPU-сборка с индекса PyTorch (без CUDA образ в разы меньше).
# mlx-whisper пропускается сам (маркер платформы: только macOS/arm64);
# gigaam ставится тарболом — git в образе не нужен.
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY app ./app

# Данные (БД, модели, образцы голосов, записи) — в томе /data.
# Внешний LLM по API (вместо Ollama) настраивается переменными STENOGRAF_LLM_*.
ENV STENOGRAF_DATA_DIR=/data
VOLUME /data

EXPOSE 8765
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]

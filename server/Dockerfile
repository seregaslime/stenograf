# Сервер Стенографа: FastAPI + GigaAM/whisper + silero-vad + ECAPA.
# Сценарий: сервер организации (Linux), клиенты подключаются по сети.
#
# Рабочий образ:   docker compose up -d --build
# Прогон тестов:   docker build --target test -t stenograf-test ./server
#                  docker run --rm stenograf-test
FROM python:3.12-slim AS base

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


# --- Стадия тестов -----------------------------------------------------------
# Отдельный образ: те же зависимости и тот же код, но плюс pytest и сами тесты.
# Смысл — доказать, что тесты и покрытие воспроизводятся на чистой Linux-машине,
# а не только на макбуке разработчика. В рабочий образ это не попадает.
FROM base AS test

COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pytest.ini .coveragerc ./
COPY tests ./tests
COPY scripts ./scripts

# Модели не качаем: медленные уровни (integration/e2e/load) требуют весов и
# живого сервера, здесь гоняется быстрый набор с покрытием.
ENV STENOGRAF_PRELOAD_ASR=false
CMD ["python", "scripts/test_report.py", "--out", "/srv/reports/TEST_REPORT.md"]


# --- Рабочий образ (стадия по умолчанию) -------------------------------------
FROM base AS runtime

# Версия приезжает аргументом сборки: без неё сервер отдавал бы в /api/health
# захардкоженную строку, одинаковую у образа от 24 августа и у сегодняшнего.
# Именно по ней видно, отстала машина или нет — «0.1.0» на этот вопрос не
# отвечает никогда.
ARG STENOGRAF_VERSION=dev
ENV STENOGRAF_VERSION=${STENOGRAF_VERSION}

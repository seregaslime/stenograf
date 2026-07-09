#!/bin/sh
# Запуск сервера Стенографа: ./run.sh
# При первом запуске сам создаёт окружение и ставит зависимости.
# Порт и адрес можно поменять переменными: PORT=9000 HOST=0.0.0.0 ./run.sh
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/uvicorn ]; then
    echo "Окружения ещё нет — создаю (займёт пару минут)..."
    if command -v uv >/dev/null 2>&1; then
        uv venv --python 3.12 .venv
        uv pip install -r requirements.txt --python .venv/bin/python
    else
        python3 -m venv .venv
        .venv/bin/pip install -r requirements.txt
    fi
fi

exec .venv/bin/uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8765}"

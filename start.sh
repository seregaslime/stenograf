#!/bin/sh
# Запускает Стенограф целиком: Ollama + сервер + приложение.
#   ./start.sh
# Сервер пишет лог в server.log. Остановить сервер: pkill -f "uvicorn app.main"
cd "$(dirname "$0")"

# 1. Ollama (локальная LLM для резюме и подсказок)
if ! curl -s --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Запускаю Ollama..."
    (ollama serve >/dev/null 2>&1 &)
fi

# 2. Сервер распознавания
if ! curl -s --max-time 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1; then
    echo "Запускаю сервер (лог: server.log)..."
    (sh server/run.sh > server.log 2>&1 &)
fi

printf "Жду сервер (первый раз может грузить модели)"
until curl -s --max-time 2 http://127.0.0.1:8765/api/health >/dev/null 2>&1; do
    printf "."
    sleep 2
done
echo " готов!"

# 3. Приложение: установленное → собранное → dev-режим
if [ -d "/Applications/Стенограф.app" ]; then
    open "/Applications/Стенограф.app"
elif [ -d "client/release/mac-arm64/Стенограф.app" ]; then
    open "client/release/mac-arm64/Стенограф.app"
else
    echo "Собранного приложения нет — запускаю dev-режим (Ctrl+C для выхода)"
    cd client && npm run dev
fi

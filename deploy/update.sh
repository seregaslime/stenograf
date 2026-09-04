#!/bin/sh
# Обновление сервера Стенографа до свежего образа из реестра.
#
# Почему тянем, а не собираем: на машине деплоя 4 ядра и меньше двух гигабайт
# свободной памяти, а сборка ставит torch. Одна неудачная сборка — и сервер
# лежит до тех пор, пока кто-нибудь не придёт руками.
#
#   ssh root@<хост> 'cd /opt/stenograf && sh deploy/update.sh'
#
# Имена переменных ЛАТИНИЦЕЙ намеренно: на Ubuntu /bin/sh — это dash, а он
# допускает в именах только ASCII. На маке /bin/sh это bash, который кириллицу
# терпит, поэтому проверка `sh -n` на ноутбуке проходила, а на сервере скрипт
# падал с «not found» на первом же присваивании.
#
# Откат: в docker-compose.override.yml прописать образ с нужным тегом-sha
# (их публикует CI на каждый мерж) и поднять сервис заново.
set -eu

dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$dir"

echo "→ Обновляю описание сервисов"
git pull --ff-only

# Копия ДО обновления и внутри тома: миграции побегут при старте нового
# контейнера, и откат образа схему БД обратно не откатит. Без этой строки
# неудачное обновление превращается в потерю встреч.
stamp=$(date +%Y%m%d-%H%M)
echo "→ Копия базы: stenograf.backup-$stamp.db"
docker compose exec -T server sh -c "cp -f /data/stenograf.db /data/stenograf.backup-$stamp.db" \
  || echo "  (базы ещё нет — первая установка)"

echo "→ Тяну образ"
docker compose pull server

echo "→ Перезапускаю"
docker compose up -d server

echo "→ Жду, пока поднимется"
health=""
tries=0
while [ "$tries" -lt 60 ]; do
    health=$(curl -fsS -m 3 http://127.0.0.1:8765/api/health 2>/dev/null || true)
    if [ -n "$health" ]; then
        break
    fi
    tries=$((tries + 1))
    sleep 2
done
if [ -z "$health" ]; then
    echo "✗ Сервер не поднялся за две минуты. Журнал: docker compose logs --tail 50 server"
    exit 1
fi
echo "  сервер отвечает: $health"

# Чистим только висячие слои: место на диске кончается за десяток обновлений
# (образ весит 2.4 ГБ), но теги с sha трогать нельзя — на них откатываются.
echo "→ Убираю висячие слои"
docker image prune -f >/dev/null

echo "✓ Готово. Версия видна в /api/health и в строке состояния приложения."

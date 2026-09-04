#!/bin/sh
# Обновление сервера Стенографа до свежего образа из реестра.
#
# Почему тянем, а не собираем: на машине деплоя 4 ядра и меньше двух гигабайт
# свободной памяти, а сборка ставит torch. Одна неудачная сборка — и сервер
# лежит до тех пор, пока кто-нибудь не придёт руками.
#
#   ssh root@<хост> 'cd /opt/stenograf && sh deploy/update.sh'
#
# Откат: docker compose down server && docker tag <sha-тег> latest && up -d,
# либо прописать нужный sha в compose. Теги с sha публикует CI на каждый мерж.
set -eu

КАТАЛОГ=$(cd "$(dirname "$0")/.." && pwd)
cd "$КАТАЛОГ"

echo "→ Обновляю описание сервисов"
git pull --ff-only

# Бэкап ДО обновления и внутри тома: миграции побегут при старте нового
# контейнера, и откат образа схему БД обратно не откатит. Без этой строки
# неудачное обновление превращается в потерю встреч.
МЕТКА=$(date +%Y%m%d-%H%M)
echo "→ Копия базы: stenograf.backup-$МЕТКА.db"
docker compose exec -T server sh -c "cp -f /data/stenograf.db /data/stenograf.backup-$МЕТКА.db" \
  || echo "  (базы ещё нет — первая установка)"

echo "→ Тяну образ"
docker compose pull server

echo "→ Перезапускаю"
docker compose up -d server

echo "→ Жду, пока поднимется"
ПОПЫТКА=0
while [ "$ПОПЫТКА" -lt 60 ]; do
    ОТВЕТ=$(curl -fsS -m 3 http://127.0.0.1:8765/api/health 2>/dev/null || true)
    if [ -n "$ОТВЕТ" ]; then
        echo "  сервер отвечает: $ОТВЕТ"
        break
    fi
    ПОПЫТКА=$((ПОПЫТКА + 1))
    sleep 2
done
if [ -z "${ОТВЕТ:-}" ]; then
    echo "✗ Сервер не поднялся за две минуты. Журнал: docker compose logs --tail 50 server"
    exit 1
fi

# Чистим только висячие слои: место на диске кончается за десяток обновлений
# (образ весит 2.4 ГБ), но теги с sha трогать нельзя — на них откатываются.
echo "→ Убираю висячие слои"
docker image prune -f >/dev/null

echo "✓ Готово. Версия видна в /api/health и в строке состояния приложения."

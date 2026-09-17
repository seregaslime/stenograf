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
# (их публикует CI на каждый мерж) и поднять сервис заново. База при откате
# восстанавливается из дампа:
#   docker compose exec -T db psql -U stenograf stenograf < backups/stenograf-<метка>.sql
set -eu

dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$dir"

echo "→ Обновляю описание сервисов"
git pull --ff-only

# Копия ДО обновления: миграции побегут при старте нового контейнера, и откат
# образа схему БД обратно не откатит. Без этой строки неудачное обновление
# превращается в потерю встреч.
#
# pg_dump, а не копия файла: база переехала в PostgreSQL, и «скопировать файл»
# у работающей базы даёт снимок, из которого она может не подняться. Дамп
# кладём на хост, а не в том — так его видно и легко забрать себе.
#
# Без конвейера с gzip намеренно: в dash нет pipefail, и упавший pg_dump с
# успешным gzip дал бы пустой файл и зелёный статус — то есть тихую потерю
# копии ровно там, где она нужна.
stamp=$(date +%Y%m%d-%H%M)
mkdir -p backups
echo "→ Копия базы: backups/stenograf-$stamp.sql"
if docker compose exec -T db pg_dump -U stenograf stenograf > "backups/stenograf-$stamp.sql"; then
    :
else
    echo "  (базы ещё нет — первая установка)"
    rm -f "backups/stenograf-$stamp.sql"
fi

# Сменившийся образ базы подменять молча нельзя: `up -d server` пересоздаёт и
# базу, от которой сервер зависит, — новый образ поверх старого тома. Так
# 17.09.2026 postgres:17-alpine сменился на pgvector/pgvector:pg17, а текст
# Alpine и Debian сортируют по-разному: индексы старого тома у нового образа
# врут. Базу сначала переносят дампом, а уже потом обновляются.
running_db=$(docker compose ps -q db 2>/dev/null || true)
if [ -n "$running_db" ]; then
    have_db=$(docker inspect -f '{{.Config.Image}}' "$running_db")
    want_db=$(docker compose config | sed -n '/^  db:$/,/^  [a-z]/p' | awk '$1 == "image:" {print $2; exit}')
    if [ "$have_db" = "$want_db" ]; then
        :
    else
        echo "✗ Образ базы сменился: $have_db → $want_db."
        echo "  Поверх старого тома его ставить нельзя — сначала перенос: sh deploy/move-db.sh"
        exit 1
    fi
fi

echo "→ Тяну образ"
docker compose pull server

echo "→ Перезапускаю"
# --remove-orphans убирает контейнеры, которых в описании больше нет. Иначе
# исчезнувший из compose сервис (так ушла ollama) продолжает работать и держать
# память: docker обновляет только то, что перечислено, а про остальное молчит.
docker compose up -d --remove-orphans server

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

#!/bin/sh
# Перенос базы на новый образ Postgres — дампом, а не подменой образа.
#
# Зачем появился: 17.09.2026 образ базы сменился с postgres:17-alpine на
# pgvector/pgvector:pg17 — поиск по встречам переехал на pgvector. Версия
# Postgres та же, но Alpine и Debian по-разному сортируют текст (musl против
# glibc), и индексы по тексту, построенные старым образом, у нового могут врать:
# запись в таблице есть, а поиск по индексу её не находит. Первым бы сломался
# вход — токен ищется по индексу. Дамп и восстановление строят индексы заново,
# уже новым образом.
#
#   ssh root@<хост> 'cd /opt/stenograf && sh deploy/move-db.sh'
#   ssh root@<хост> 'cd /opt/stenograf && sh deploy/update.sh'
#
# update.sh сам откажется обновляться, пока образ базы в описании и у
# работающей базы разный, и пришлёт сюда.
#
# Откат: старый том не удаляется бесследно, а копируется в <том>-before-<метка>.
# Вернуть прежний образ в docker-compose.yml и смонтировать эту копию — или
# восстановить дамп из backups/move-<метка>.sql в базу на прежнем образе.
#
# Имена переменных латиницей — по той же причине, что в update.sh: на Ubuntu
# /bin/sh это dash.
set -eu

dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$dir"

# Образ базы по описанию сервисов — с учётом docker-compose.override.yml
db_image() {
    docker compose config | sed -n '/^  db:$/,/^  [a-z]/p' | awk '$1 == "image:" {print $2; exit}'
}

running=$(docker compose ps -q db)
if [ -z "$running" ]; then
    echo "✗ База не запущена — снимать дамп не с чего."
    echo "  Поднимите её на прежнем образе и запустите перенос снова."
    exit 1
fi
have=$(docker inspect -f '{{.Config.Image}}' "$running")
want=$(db_image)
if [ "$have" = "$want" ]; then
    echo "Образ базы уже $want — переносить нечего."
    exit 0
fi

stamp=$(date +%Y%m%d-%H%M)
mkdir -p backups
dump="backups/move-$stamp.sql"
echo "→ Дамп базы на $have: $dump"
docker compose exec -T db pg_dump -U stenograf stenograf > "$dump"
# Пустой дамп при нулевом коде — и дальше мы удалили бы том, из которого его
# снимали. Проверка дешевле, чем потерянный архив встреч.
if [ ! -s "$dump" ]; then
    echo "✗ Дамп пустой — останавливаюсь, база не тронута."
    exit 1
fi

volume=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql/data"}}{{.Name}}{{end}}{{end}}' "$running")
echo "→ Останавливаю сервер и базу"
docker compose stop server db
docker compose rm -f db

echo "→ Копия старого тома для отката: $volume-before-$stamp"
docker volume create "$volume-before-$stamp" >/dev/null
docker run --rm -v "$volume":/from -v "$volume-before-$stamp":/to "$have" sh -c 'cp -a /from/. /to/'
docker volume rm "$volume" >/dev/null

echo "→ Новая база на $want"
docker compose pull db
docker compose up -d db
# Ждём по TCP, а не по сокету: при первом запуске образ поднимает временный
# сервер без сети, создаёт базу и перезапускается. pg_isready по сокету сказал бы
# «готово» временному серверу, и дамп полетел бы в базу посреди перезапуска.
tries=0
until docker compose exec -T db pg_isready -h 127.0.0.1 -U stenograf >/dev/null 2>&1; do
    tries=$((tries + 1))
    if [ "$tries" -ge 60 ]; then
        echo "✗ Новая база не поднялась за две минуты. Журнал: docker compose logs --tail 50 db"
        echo "  Дамп цел: $dump, копия старого тома: $volume-before-$stamp"
        exit 1
    fi
    sleep 2
done

echo "→ Восстанавливаю дамп"
# Вывод запросов не нужен (дамп печатает по строке на каждую последовательность),
# а ошибки идут в stderr и остаются видны.
docker compose exec -T db psql -q -v ON_ERROR_STOP=1 -U stenograf stenograf < "$dump" > /dev/null

echo "✓ База перенесена на $want. Дальше: sh deploy/update.sh"
echo "  Когда всё проверено, копию можно удалить: docker volume rm $volume-before-$stamp"

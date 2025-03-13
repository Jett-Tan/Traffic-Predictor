docker compose up airflow-init
docker compose up

# Rebuild

docker compose down
docker compose build
docker compose up -d

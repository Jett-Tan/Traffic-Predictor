"""
Shared Postgres connection helper for all DAGs.
Keeps the connection string in one place.
"""
import psycopg2


def get_postgres_conn():
    """
    Return a psycopg2 connection to the Postgres service used by Airflow.
    Adjust the credentials only if your docker‑compose.yml is different.
    """
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow",
    )
    return conn

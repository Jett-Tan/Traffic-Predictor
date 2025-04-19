"""
Utility: central Postgres connection helper used by all DAGs.
Placed under dags/utils so Airflow can import it without registering a DAG.
"""
import psycopg2

def get_postgres_conn():
    return psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow",
    )

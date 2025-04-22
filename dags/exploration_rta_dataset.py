from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
CSV_FILE_PATH_RAW = "/opt/airflow/dags/data/RTA_Dataset.csv"
CSV_FILE_PATH_EXPLORATION = "/opt/airflow/dags/data/RTA_Dataset_Cleaned.csv"

# Postgres Connection
POSTGRES_CONN_ID = "postgres_default"
TABLE_CLEAN = "clean_rta"

def clean_data():
    df = pd.read_csv(CSV_FILE_PATH_RAW)
    
    

def exploratory_analysis():
    df = pd.read_csv(CSV_FILE_PATH_EXPLORATION)
    
    # Summary Statistics
    print(df.describe(include='all'))
    

with DAG(
    dag_id='exploration_rta_dataset',
    start_date=datetime(2025, 3, 24),
    schedule_interval='@once',
    catchup=False,
    tags=['rta', 'data_cleaning', 'data_exploration',"IS3107"]
) as dag:
    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )
    
    explore_task = PythonOperator(
        task_id='exploratory_analysis',
        python_callable=exploratory_analysis,
    )
    
    clean_task >> explore_task

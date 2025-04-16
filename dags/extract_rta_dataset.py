from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import kaggle

# Define constants
KAGGLE_DATASET = 'saurabhshahane/road-traffic-accidents'
FILE_TO_EXTRACT = 'RTA Dataset.csv'
DOWNLOAD_DIR = '/tmp/kaggle_rta_dataset'

# Python function to download the dataset
def download_rta_dataset():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_file(
        dataset=KAGGLE_DATASET,
        file_name=FILE_TO_EXTRACT,
        path=DOWNLOAD_DIR,
        force=True
    )
    print(f"File downloaded to {DOWNLOAD_DIR}")

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

with DAG(
    dag_id='extract_rta_dataset',
    default_args=default_args,
    schedule_interval='@once',  
    catchup=False,  # Prevents backfill
    description='DAG to extract RTA Dataset from Kaggle',
    tags=['kaggle', 'rta', 'traffic', 'extract',"IS3107"],
) as dag:

    download_task = PythonOperator(
        task_id='download_rta_dataset',
        python_callable=download_rta_dataset,
    )

    download_task
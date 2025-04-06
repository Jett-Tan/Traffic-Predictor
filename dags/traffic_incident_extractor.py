from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import os
import json

# Constants
LTA_API_URL = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
API_KEY = "tLB/oJ67QO+OA992i/dU7Q=="
SAVE_DIR = "/opt/airflow/data"

def fetch_lta_data():
    headers = {"AccountKey": API_KEY, "accept": "application/json"}
    response = requests.get(LTA_API_URL, headers=headers)

    if response.status_code == 200:
        data = response.json()

        # Ensure target directory exists
        os.makedirs(SAVE_DIR, exist_ok=True)

        filename = os.path.join(
            SAVE_DIR,
            f"lta_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        print(f"✅ Data saved to {filename}")
    else:
        print(f"❌ Failed to fetch data: {response.status_code}")


# Define DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "lta_datamall_pull",
    default_args=default_args,
    description="Pulls data from LTA DataMall every 1 hour and saves to a folder",
    schedule_interval="0 * * * *",  # Every hour
    catchup=False,
)

# Define task
fetch_task = PythonOperator(
    task_id="fetch_lta_data",
    python_callable=fetch_lta_data,
    dag=dag,
)

fetch_task

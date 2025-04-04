from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import pandas as pd

def get_latest_rainfall():
    # === Step 1: Fetch data from API ===
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error:", response.status_code)
        return

    data = response.json()

    # === Step 2: Extract timestamp and readings ===
    timestamp = data["items"][0]["timestamp"]
    readings = data["items"][0]["readings"]
    stations = {s["id"]: s for s in data["metadata"]["stations"]}

    # === Step 3: Combine readings with metadata ===
    combined = []
    for r in readings:
        station = stations.get(r["station_id"], {})
        combined.append({
            "Station": station.get("name", r["station_id"]),
            "Latitude": station.get("location", {}).get("latitude"),
            "Longitude": station.get("location", {}).get("longitude"),
            "Rainfall (mm)": r["value"],
            "Collected_At": timestamp
        })

    df = pd.DataFrame(combined)

    # === Step 4: Save to timestamped CSV ===
    now = datetime.now().strftime("%Y%m%d_%H%M")
    now = datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = f"/Users/ruoqiili/Desktop/is3107_dags/Traffic-Predictor/dags/data/rainfall/rainfall_data_{now}.csv"

    df.to_csv(csv_filename, index=False)

    # === Step 5: Preview result ===
    print(f"Collected {len(df)} rainfall readings at {timestamp}")
    print(f"Saved to {csv_filename}")
    print(df.head())

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 29),
}

with DAG(
    dag_id='daily_rainfall',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
    tags=['rainfall', 'data_collection']
) as dag:
    fetch_rainfall = PythonOperator(
        task_id='get_latest_rainfall',
        python_callable=get_latest_rainfall
    )

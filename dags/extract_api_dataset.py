import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2

API_KEY = "bwU/8aykSlmp6onSTQ8w0Q=="

def get_live_traffic_incidents():
    # === CONFIG ===
    URL = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
    
    # === HEADERS ===
    headers = {
        "AccountKey": API_KEY,
        "accept": "application/json"
    }

    # === FETCH DATA ===
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()  # Will raise an exception for 4xx/5xx HTTP errors
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from LTA API: {e}")
        raise  # Raise exception to notify Airflow of failure

    data = response.json()
    incidents = data["value"]

    # === CONVERT TO DATAFRAME ===
    df = pd.DataFrame(incidents)

    # === SHOW RESULT ===
    print(df[["Type", "Latitude", "Longitude", "Message"]].head())

    # === OPTIONAL: SAVE TO FILE ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/opt/airflow/dags/data/live_traffic_incidents_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

def get_weather_data():
    
    # === Step 1: Fetch data from API ===
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error:", response.status_code)
        exit()

    data = response.json()

    # === Step 2: Extract timestamp and readings ===
    timestamp = data["items"][0]["timestamp"]  # ISO string like "2025-03-29T20:45:00+08:00"
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
            "Collected_At": timestamp  # Add timestamp to each row
        })

    df = pd.DataFrame(combined)

    # === Step 4: Save to timestamped CSV ===
    now = datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = f"/opt/airflow/dags/data/rainfall_data_{now}.csv"
    df.to_csv(csv_filename, index=False)

    # === Step 5: Preview result ===
    print(f"✅ Collected {len(df)} rainfall readings at {timestamp}")
    print(f"📄 Saved to {csv_filename}")
    print(df.head())

def get_postgres_conn():
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )
    return conn


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

# schedule_interval='@hourly',  # Run every hour
with DAG(
    dag_id='extract_api_dataset',
    default_args=default_args,
    schedule_interval='*/5 * * * *',
    catchup=False,  # Prevents backfill
    description='fetch live traffic incidents from LTA API',
    tags=['rta', 'traffic', 'api'],
) as dag:
    
    get_live_traffic_incidents_python = PythonOperator(
        task_id='get_live_traffic_incidents',
        python_callable=get_live_traffic_incidents,
    )
    get_weather_data_python = PythonOperator(
        task_id='get_weather_data',
        python_callable=get_weather_data,
    )

    get_live_traffic_incidents_python >> get_weather_data_python
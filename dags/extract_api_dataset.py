import requests
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2
import os

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # === CONVERT TO DATAFRAME ===
    df = pd.DataFrame(incidents)
    df["collected_at"]= timestamp
    
    # === RENAME COLUMNS ===
    df.rename(columns={
        "Type": "type",
        "Latitude": "latitude",
        "Longitude": "longitude",
        "Message": "message",
    }, inplace=True)
    # === SHOW RESULT ===
    print(df[["type", "latitude", "longitude", "message"]].head())

    # === OPTIONAL: SAVE TO FILE ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/opt/airflow/dags/data/live_traffic/live_traffic_incidents_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

    # === SAVE TO POSTGRES ===
    save_to_postgres_live_traffic(df)
    print(f"Saved to PostgreSQL table live_traffic_incidents")

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
            "station": station.get("name", r["station_id"]),
            "latitude": station.get("location", {}).get("latitude"),
            "longitude": station.get("location", {}).get("longitude"),
            "rainfall": r["value"],
            "collected_at": timestamp  # Add timestamp to each row
        })

    df = pd.DataFrame(combined)

    # === Step 4: Save to timestamped CSV ===
    now = datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = f"/opt/airflow/dags/data/rainfall/rainfall_data_{now}.csv"
    df.to_csv(csv_filename, index=False)

    # === Step 5: Preview result ===
    print(f"✅ Collected {len(df)} rainfall readings at {timestamp}")
    print(f"📄 Saved to {csv_filename}")
    print(df.head())

    # === Step 6: Save to PostgreSQL ===
    save_to_postgres_rainfall(df)
    print(f"📦 Saved to PostgreSQL table rainfall_data")

def get_postgres_conn():
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )
    return conn

def save_to_postgres_rainfall(df):
    conn = get_postgres_conn()
    cursor = conn.cursor()
    table_name = "rainfall_data"
    # Create table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            station VARCHAR(255),
            latitude FLOAT,
            longitude FLOAT,
            rainfall FLOAT,
            collected_at TIMESTAMP
        )
    """)

    # Insert data into the table
    for _, row in df.iterrows():
        hasExist = cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1 FROM {table_name} WHERE station = %s AND latitude = %s AND longitude = %s AND rainfall = %s AND collected_at = %s
            )""", (row['station'], row['latitude'], row['longitude'], row['rainfall'], row['collected_at']))
        # Check if the record already exists
        hasExist = cursor.fetchone()[0]
        if hasExist:
            print(f"Record already exists in {table_name}: {row['station']}, {row['latitude']}, {row['longitude']}, {row['rainfall']}, {row['collected_at']}")
            continue
        cursor.execute(f"""
            INSERT INTO {table_name} (station, latitude, longitude, rainfall, collected_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (row['station'], row['latitude'], row['longitude'], row['rainfall'], row['collected_at']))

    conn.commit()
    cursor.close()
    conn.close()

def save_to_postgres_live_traffic(df):
    conn = get_postgres_conn()
    cursor = conn.cursor()
    table_name = "live_traffic_incidents"
    # Create table if it doesn't exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            type VARCHAR(255),
            latitude FLOAT,
            longitude FLOAT,
            message TEXT
        )
    """)

    # Insert data into the table
    for _, row in df.iterrows():
        hasExist = cursor.execute(f"""
            SELECT EXISTS (
                SELECT 1 FROM {table_name} WHERE type = %s AND latitude = %s AND longitude = %s AND message = %s
            )
        """, (row['type'], row['latitude'], row['longitude'], row['message']))
        # Check if the record already exists
        hasExist = cursor.fetchone()[0]
        if hasExist:
            print(f"Record already exists in {table_name}: {row['type']}, {row['latitude']}, {row['longitude']}, {row['message']}")
            continue

        cursor.execute(f"""
            INSERT INTO {table_name} (type, latitude, longitude, message)
            VALUES (%s, %s, %s, %s)
        """, (row['type'], row['latitude'], row['longitude'], row['message']))

    conn.commit()
    cursor.close()
    conn.close()

def upload_csv_to_postgres():
    # Read the CSV file
    folder_path = "/opt/airflow/dags/data/rainfall"
    # Get the latest CSV file in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_file_path)

        # Save to PostgreSQL
        save_to_postgres_rainfall(df)
        print(f"📦 Saved to PostgreSQL table rainfall_data")

    folder_path = "/opt/airflow/dags/data/live_traffic"
    # Get the latest CSV file in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    for csv_file in csv_files:
        csv_file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(csv_file_path)

        # Save to PostgreSQL
        save_to_postgres_live_traffic(df)
        print(f"📦 Saved to PostgreSQL table live_traffic_incidents")

def postgres_to_csv():
    # Connect to PostgreSQL
    conn = get_postgres_conn()
    cursor = conn.cursor()

    # Fetch data from PostgreSQL for the table 'rainfall_data'
    cursor.execute("SELECT * FROM rainfall_data")
    rows = cursor.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

    # Save to CSV
    csv_file_path = "/opt/airflow/dags/data/rainfall/rainfall_data_postgres.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"📦 Saved data to {csv_file_path}")

    # Fetch data from PostgreSQL for the table 'live_traffic_incidents'
    cursor.execute("SELECT * FROM live_traffic_incidents")
    rows = cursor.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(rows, columns=[desc[0] for desc in cursor.description])

    # Save to CSV
    csv_file_path = "/opt/airflow/dags/data/live_traffic/live_traffic_incidents_postgres.csv"
    df.to_csv(csv_file_path, index=False)
    print(f"📦 Saved data to {csv_file_path}")

    # Close the connection
    cursor.close()
    conn.close()

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

with DAG(
    dag_id='extract_api_dataset',
    default_args=default_args,
    # schedule_interval='*/5 * * * *',
    schedule_interval='@hourly',  # Run every hour
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

with DAG(
    dag_id='uploadCSVtoPostgres',
    default_args=default_args,
    # schedule_interval='*/5 * * * *',
    schedule_interval='@once',  # Run once
    catchup=False,  # Prevents backfill
    description='fetch live traffic incidents from LTA API',
    tags=['rta', 'traffic', 'api'],
) as dag:
    
    upload_csv_to_postgres_python = PythonOperator(
        task_id='upload_csv_to_postgres',
        python_callable=upload_csv_to_postgres,
    )
    postgres_to_csv_python = PythonOperator(
        task_id='postgres_to_csv',
        python_callable=postgres_to_csv,
    )
    

    upload_csv_to_postgres_python >> postgres_to_csv_python


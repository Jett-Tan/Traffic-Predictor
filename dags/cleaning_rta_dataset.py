from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt

from load_rta_dataset import get_postgres_conn

# Paths
CSV_FILE_PATH_RAW = "/opt/airflow/dags/data/RTA_Dataset.csv"
CSV_FILE_PATH_CLEANED = "/opt/airflow/dags/data/RTA_Dataset_Cleaned.csv"

# Postgres Connection
POSTGRES_CONN_ID = "postgres_default"
TABLE_NAME_CLEANED = "cleaned_rta"

def create_cleaned_RTA_table():
    # Database connection
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cleaned_rta (
            id SERIAL PRIMARY KEY,
            day_of_week VARCHAR,
            age_band_of_driver VARCHAR,
            sex_of_driver VARCHAR,
            educational_level VARCHAR,
            vehicle_driver_relation VARCHAR,
            driving_experience VARCHAR,
            type_of_vehicle VARCHAR,
            owner_of_vehicle VARCHAR,
            area_accident_occured VARCHAR,
            lanes_or_medians VARCHAR,
            road_alignment VARCHAR,
            types_of_junction VARCHAR,
            road_surface_conditions VARCHAR,
            light_conditions VARCHAR,
            weather_conditions VARCHAR,
            type_of_collision VARCHAR,
            number_of_vehicles_involved INT,
            number_of_casualties INT,
            vehicle_movement VARCHAR,
            age_band_of_casualty VARCHAR,
            casualty_severity VARCHAR,
            cause_of_accident VARCHAR,
            accident_severity VARCHAR,
            hour INT
        );
    """)
    conn.commit()
    cursor.close()
    conn.close()

def drop_cleaned_RTA_table():
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DROP TABLE IF EXISTS {TABLE_NAME_CLEANED};
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_cleaned_data_into_postgres():
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH_CLEANED)
    print(len(df.columns))
    
    conn = get_postgres_conn()
    cursor = conn.cursor()

    # Ensure we're not inserting 'id'
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    # Insert data
    for _, row in df.iterrows():
        print("Row:", row)
        print("Length of row:", len(row))
        print("Tuple:", tuple(row))
        cursor.execute("""
            INSERT INTO cleaned_rta (
                day_of_week, age_band_of_driver, sex_of_driver,
                educational_level, vehicle_driver_relation, driving_experience,
                type_of_vehicle, owner_of_vehicle, area_accident_occured,
                lanes_or_medians, road_alignment, types_of_junction,
                road_surface_conditions, light_conditions, weather_conditions,
                type_of_collision, number_of_vehicles_involved, number_of_casualties,
                vehicle_movement, age_band_of_casualty, casualty_severity,
                cause_of_accident, accident_severity, hour
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

def clean_data():
    conn = get_postgres_conn()
    df = pd.read_sql("SELECT * FROM rta", conn)
    print("RAW", df.info())
    
    # Remove defect_of_vehicle, service_year_of_vehicle, work_of_casuality, fitness_of_casuality
    df_cleaned = df.drop(columns=['defect_of_vehicle', 'service_year_of_vehicle', 'work_of_casuality', 'fitness_of_casuality'])
    # Standardize column values (lowercase, strip spaces, replace underscores)
    df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(" ", "_")

    # Rename the spelling error
    df_cleaned.rename(columns={"road_allignment": "road_alignment"}, inplace=True)

    # Group into hour
    df_cleaned["time"] = pd.to_datetime(df_cleaned["time"], format="%H:%M:%S")
    df_cleaned["hour"] = df_cleaned["time"].dt.hour # group by the hour

    # There are many NA values in casualty_class and sex_of_casualty. 
    # Since these are not directly related to the goal of our prediction, we can drop these columns 
    df_cleaned.drop(columns=["casualty_class", "sex_of_casualty"], inplace=True)

    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

    # Further drop irrelevant columns
    df_cleaned.drop(columns=["pedestrian_movement", "road_surface_type", "time"], inplace=True)

    # Strip whitespace from all string columns
    df_cleaned = df_cleaned.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

    print(df_cleaned.info())

    df_cleaned.to_csv(CSV_FILE_PATH_CLEANED, index=False)

with DAG(
    dag_id='cleaning_rta_dataset',
    start_date=datetime(2025, 3, 24),
    schedule_interval='@once',
    catchup=False,
    tags=['rta', 'data_cleaning', "IS3107"],
) as dag:
    drop_cleaned_table_task = PythonOperator(
        task_id='drop_cleaned_RTA_table',
        python_callable=drop_cleaned_RTA_table,
    )

    create_cleaned_table_task = PythonOperator(
        task_id='create_cleaned_RTA_table',
        python_callable=create_cleaned_RTA_table,
    )

    insert_cleaned_data_task = PythonOperator(
        task_id='insert_cleaned_data_into_postgres',
        python_callable=insert_cleaned_data_into_postgres,
    )

    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )
    
    # Task dependencies
    drop_cleaned_table_task >> create_cleaned_table_task >> clean_data_task >> insert_cleaned_data_task

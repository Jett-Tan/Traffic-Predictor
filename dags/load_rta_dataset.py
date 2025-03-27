from airflow import DAG
from airflow.operators.python import PythonOperator
# from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime, timedelta
import pandas as pd
import psycopg2

# PostgreSQL Connection Details
POSTGRES_CONN_ID = "postgres_default"

# Path to CSV file (inside the Airflow container)
CSV_FILE_PATH = "/opt/airflow/dags/data/RTA_Dataset.csv"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

def create_RTA_table():
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)  # Connection must exist in Airflow UI
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rta (
            id SERIAL PRIMARY KEY,
            Time TIME,
            Day_of_week VARCHAR,
            Age_band_of_driver VARCHAR,
            Sex_of_driver VARCHAR,
            Educational_level VARCHAR,
            Vehicle_driver_relation VARCHAR,
            Driving_experience VARCHAR,
            Type_of_vehicle VARCHAR,
            Owner_of_vehicle VARCHAR,
            Service_year_of_vehicle VARCHAR,
            Defect_of_vehicle VARCHAR,
            Area_accident_occured VARCHAR,
            Lanes_or_Medians VARCHAR,
            Road_allignment VARCHAR,
            Types_of_Junction VARCHAR,
            Road_surface_type VARCHAR,
            Road_surface_conditions VARCHAR,
            Light_conditions VARCHAR,
            Weather_conditions VARCHAR,
            Type_of_collision VARCHAR,
            Number_of_vehicles_involved INT,
            Number_of_casualties INT,
            Vehicle_movement VARCHAR,
            Casualty_class VARCHAR,
            Sex_of_casualty VARCHAR,
            Age_band_of_casualty VARCHAR,
            Casualty_severity VARCHAR,
            Work_of_casuality VARCHAR,
            Fitness_of_casuality VARCHAR,
            Pedestrian_movement VARCHAR,
            Cause_of_accident VARCHAR,
            Accident_severity VARCHAR
        );  
    """)
    conn.commit()
    cursor.close()
    conn.close()

def drop_RTA_table():
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)  # Connection must exist in Airflow UI
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        DROP TABLE IF EXISTS rta;
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_data_into_postgres():
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH)

    # Database connection
    conn = psycopg2.connect(
        host="postgres",  # Container name of PostgreSQL
        database="airflow",
        user="airflow",
        password="airflow"
    )
    cur = conn.cursor()

    # Insert data
    for _, row in df.iterrows():
        cur.execute("""
            INSERT INTO rta (
                Time, Day_of_week, Age_band_of_driver, Sex_of_driver,
                Educational_level, Vehicle_driver_relation, Driving_experience,
                Type_of_vehicle, Owner_of_vehicle, Service_year_of_vehicle,
                Defect_of_vehicle, Area_accident_occured, Lanes_or_Medians,
                Road_allignment, Types_of_Junction, Road_surface_type,
                Road_surface_conditions, Light_conditions, Weather_conditions,
                Type_of_collision, Number_of_vehicles_involved, Number_of_casualties,
                Vehicle_movement, Casualty_class, Sex_of_casualty, Age_band_of_casualty,
                Casualty_severity, Work_of_casuality, Fitness_of_casuality,
                Pedestrian_movement, Cause_of_accident, Accident_severity
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, tuple(row))

    conn.commit()
    cur.close()
    conn.close()

with DAG(
    dag_id='extract_rta_dataset',
    default_args=default_args,
    schedule_interval='@once',  
    catchup=False,  # Prevents backfill
    description='DAG to load RTA Dataset to postgres db',
    tags=['rta', 'traffic', 'postgres'],
) as dag:
    drop_RTA_table = PythonOperator(
        task_id='drop_table_python',
        python_callable=drop_RTA_table,
        dag=dag,
    )
    
    create_RTA_table = PythonOperator(
        task_id='create_table_python',
        python_callable=create_RTA_table,
        dag=dag,
    )

    load_data = PythonOperator(
        task_id="load_csv_to_postgres",
        python_callable=insert_data_into_postgres,
        dag=dag,
    )

    # create_pet_table >> populate_RTA_table
    drop_RTA_table>>create_RTA_table >> load_data
    



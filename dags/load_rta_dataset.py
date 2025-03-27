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
            Day_of_week VARCHAR(20),
            Age_band_of_driver VARCHAR(20),
            Sex_of_driver VARCHAR(10),
            Educational_level VARCHAR(50),
            Vehicle_driver_relation VARCHAR(50),
            Driving_experience VARCHAR(20),
            Type_of_vehicle VARCHAR(50),
            Owner_of_vehicle VARCHAR(50),
            Service_year_of_vehicle VARCHAR(20),
            Defect_of_vehicle VARCHAR(50),
            Area_accident_occured VARCHAR(50),
            Lanes_or_Medians VARCHAR(50),
            Road_allignment VARCHAR(50),
            Types_of_Junction VARCHAR(50),
            Road_surface_type VARCHAR(50),
            Road_surface_conditions VARCHAR(50),
            Light_conditions VARCHAR(50),
            Weather_conditions VARCHAR(50),
            Type_of_collision VARCHAR(50),
            Number_of_vehicles_involved INT,
            Number_of_casualties INT,
            Vehicle_movement VARCHAR(50),
            Casualty_class VARCHAR(50),
            Sex_of_casualty VARCHAR(10),
            Age_band_of_casualty VARCHAR(20),
            Casualty_severity VARCHAR(20),
            Work_of_casuality VARCHAR(50),
            Fitness_of_casuality VARCHAR(50),
            Pedestrian_movement VARCHAR(50),
            Cause_of_accident VARCHAR(100),
            Accident_severity VARCHAR(50)
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
    



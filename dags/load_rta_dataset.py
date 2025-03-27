from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
# PostgreSQL Connection Details
POSTGRES_CONN_ID = "postgres_default"

# Path to CSV file (inside the Airflow container)
CSV_FILE_PATH_RTA = "/opt/airflow/dags/data/RTA_Dataset.csv"
CSV_FILE_PATH_CLEAN = "/opt/airflow/dags/data/cleaned.csv"


TABLE_NAME_RTA = "rta"
TABLE_NAME_CLEAN = "clean_rta"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

def create_RTA_table_Python():
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME_RTA} (
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

def drop_RTA_table_Python():    
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DROP TABLE IF EXISTS {TABLE_NAME_RTA};
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_RTA_data_into_postgres():
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH_RTA)

   
    conn = get_postgres_conn()
    cur = conn.cursor()

    # Insert data
    for _, row in df.iterrows():
        cur.execute(f"""
            INSERT INTO {TABLE_NAME_RTA} (
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

def get_postgres_conn():
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )
    return conn

def create_clean_RTA_table():
    # Database connection
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME_CLEAN} (
            id SERIAL PRIMARY KEY,
            Age_band_of_driver VARCHAR,
            Sex_of_driver VARCHAR,
            Educational_level VARCHAR,
            Vehicle_driver_relation VARCHAR,
            Driving_experience VARCHAR,
            Lanes_or_Medians VARCHAR,
            Types_of_Junction VARCHAR,
            Road_surface_type VARCHAR,
            Light_conditions VARCHAR,
            Weather_conditions VARCHAR,
            Type_of_collision VARCHAR,
            Vehicle_movement VARCHAR,
            Pedestrian_movement VARCHAR,
            Cause_of_accident VARCHAR,
            Accident_severity INT
        );  
    """)
    conn.commit()
    cursor.close()
    conn.close()

def drop_clean_RTA_table():
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DROP TABLE IF EXISTS {TABLE_NAME_CLEAN};
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_clean_data_into_postgres():
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH_CLEAN)

    
    conn = get_postgres_conn()
    cursor = conn.cursor()

    # Insert data
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME_CLEAN} (
                Age_band_of_driver,Sex_of_driver,Educational_level,Vehicle_driver_relation,Driving_experience,Lanes_or_Medians,Types_of_Junction,Road_surface_type,Light_conditions,Weather_conditions,Type_of_collision,Vehicle_movement,Pedestrian_movement,Cause_of_accident,Accident_severity
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, tuple(row))

    conn.commit()
    cursor.close()
    conn.close()

def print_postgres_data(table_name):
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM {table_name};
    """)
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    cursor.close()
    conn.close()

with DAG(
    dag_id='postgres_rta_dataset',
    default_args=default_args,
    schedule_interval='@once',  
    catchup=False,  # Prevents backfill
    description='DAG to load RTA Dataset to postgres db',
    tags=['rta', 'traffic', 'postgres'],
) as dag:
    drop_RTA_table = PythonOperator(
        task_id='drop_RTA_table_python',
        python_callable=drop_RTA_table_Python,
        # dag=dag,
    )
    
    create_RTA_table = PythonOperator(
        task_id='create_RTA_table_python',
        python_callable=create_RTA_table_Python,
        # dag=dag,
    )

    load_RTA_data = PythonOperator(
        task_id="load_RTA_csv_to_postgres",
        python_callable=insert_RTA_data_into_postgres,
        # dag=dag,
    )
    drop_CLEAN_table = PythonOperator(
        task_id='drop_clean_RTA_table_python',
        python_callable=drop_clean_RTA_table,
        # dag=dag,
    )
    
    create_CLEAN_table = PythonOperator(
        task_id='create_clean_RTA_table_python',
        python_callable=create_clean_RTA_table,
        # dag=dag,
    )

    load_CLEAN_data = PythonOperator(
        task_id="load_clean_RTA_csv_to_postgres",
        python_callable=insert_clean_data_into_postgres,
        # dag=dag,
    )

    print_CLEAN_data = PythonOperator(
        task_id="print_clean_data",
        python_callable=print_postgres_data,
        op_args=[TABLE_NAME_CLEAN],
        # dag=dag,
    )

    drop_RTA_table >> create_RTA_table >> load_RTA_data
    drop_CLEAN_table >> create_CLEAN_table >> load_CLEAN_data
    load_RTA_data >> load_CLEAN_data >> print_CLEAN_data
    



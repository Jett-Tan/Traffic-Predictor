"""
Example Airflow DAG for data visualization from the 'rta' table in Postgres.
Generates multiple plots as PNG files.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------------------------------
# 1. REUSE/RECREATE THE POSTGRES CONNECTION LOGIC
# -----------------------------------------------------------------------------
def get_postgres_conn():
    """
    Creates a psycopg2 connection to the same Postgres DB used in load_rta_dataset.py.
    Adjust host, db, user, password if your config differs.
    """
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )
    return conn

# -----------------------------------------------------------------------------
# 2. MAIN VISUALIZATION FUNCTION
# -----------------------------------------------------------------------------
def visualize_rta_data():
    """
    Connect to Postgres, read the 'rta' table, create multiple plots, 
    and save them as PNG files.
    """
    # Ensure we have a directory to save plots
    plots_dir = "/opt/airflow/dags/data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # ---------------------------
    # Connect & Read Data
    # ---------------------------
    conn = get_postgres_conn()
    query = "SELECT * FROM rta;"  # or "SELECT * FROM clean_rta;"
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 1) ACCIDENT SEVERITY DISTRIBUTION (Bar Chart)
    plt.figure()
    severity_counts = df["Accident_severity"].value_counts(dropna=False)
    severity_counts.plot(kind='bar')
    plt.title('Accident Severity Distribution')
    plt.xlabel('Accident Severity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/accident_severity_distribution.png")
    plt.close()
    
    # 2) DAY OF WEEK VS NUMBER OF ACCIDENTS (Bar Chart)
    # Some rows may have missing or blank days; handle carefully.
    plt.figure()
    day_counts = df["Day_of_week"].value_counts(dropna=False)
    day_counts.plot(kind='bar')
    plt.title('Accidents by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/accidents_by_day_of_week.png")
    plt.close()
    
    # 3) DRIVER AGE BAND vs. ACCIDENT SEVERITY (Grouped Bar)
    # We'll pivot the data to see how many accidents of each severity occur for each age band.
    # Note: If 'Accident_severity' is sometimes text (Slight Injury, Serious Injury, etc.),
    # or sometimes numeric, you may want to standardize that first.
    
    # Step A: We assume Accident_severity might be text, let's just do a pivot of counts
    # (If your severity is "Slight Injury" / "Serious Injury", the bar chart might be large.)
    pivot_df = df.pivot_table(
        index="Age_band_of_driver", 
        columns="Accident_severity", 
        values="Time",  # any column to count
        aggfunc="count",
        fill_value=0
    )
    
    # Step B: Plot a grouped bar
    plt.figure()
    pivot_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Accident Severity by Age Band of Driver")
    plt.xlabel("Age Band of Driver")
    plt.ylabel("Count of Accidents")
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/ageband_vs_severity.png")
    plt.close()
    
    # 4) WEATHER CONDITIONS FREQUENCY (Pie Chart)
    # You could also do a bar chart. We'll do a pie for variety.
    # Some rows might have missing or blank weather conditions.
    weather_counts = df["Weather_conditions"].value_counts(dropna=False)
    plt.figure()
    weather_counts.plot(kind='pie', autopct='%1.1f%%')  # show percentages
    plt.title('Weather Conditions at Time of Accidents')
    plt.ylabel('')  # hide the y-label "Weather_conditions"
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/weather_conditions_pie.png")
    plt.close()
    
    print("Visualization PNGs saved in:", plots_dir)

# -----------------------------------------------------------------------------
# 3. DEFINE THE DAG
# -----------------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

with DAG(
    dag_id='visualize_data',
    default_args=default_args,
    schedule_interval='@once',  # or '0 12 * * *' for daily at noon, etc.
    catchup=False,
    description='DAG to generate data visualizations from the RTA dataset',
    tags=['rta', 'visualization']
) as dag:

    visualize_task = PythonOperator(
        task_id='visualize_rta_data',
        python_callable=visualize_rta_data,
    )

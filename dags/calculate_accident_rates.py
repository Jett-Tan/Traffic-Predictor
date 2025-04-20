from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
import pandas as pd
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 4, 20),
}

with DAG("calculate_accident_rates",
         default_args=default_args,
         schedule_interval="@daily",
         catchup=False,
         tags=["traffic", "analytics", "IS3107"]) as dag:

    def compute_accident_rates():
        # 1. Load the RTA dataset
        file_path = "/opt/airflow/dags/data/RTA_Dataset.csv"
        df = pd.read_csv(file_path)

        # 2. Drop rows with missing important features
        df = df.dropna(subset=["Types_of_Junction"])

        # 3. Clean feature: convert to lowercase for consistency
        df["Types_of_Junction"] = df["Types_of_Junction"].str.strip().str.lower()

        # 4. Count number of accidents per junction type
        accident_counts = df.groupby("Types_of_Junction").size().reset_index(name="accident_count")

        # 5. Normalize: accident rate proxy = count / total
        total_accidents = accident_counts["accident_count"].sum()
        accident_counts["accident_rate_proxy"] = accident_counts["accident_count"] / total_accidents

        # 6. Save to CSV for use in dashboards/visualization
        output_path = "/opt/airflow/dags/data/junction_accident_rates.csv"
        accident_counts.to_csv(output_path, index=False)

    compute_rates = PythonOperator(
        task_id="compute_accident_rates",
        python_callable=compute_accident_rates,
    )

    compute_rates

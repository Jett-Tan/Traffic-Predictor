from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from utils.db import get_postgres_conn

# Paths
CSV_FILE_PATH_CLEANED = "/opt/airflow/dags/data/RTA_Dataset_Cleaned.csv"
CSV_FILE_PATH_ENCODED = "/opt/airflow/dags/data/RTA_Dataset_Encoded.csv"
FEATURE_IMP_CSV = "/opt/airflow/dags/data/feature_importance.csv"

# Paths for storing models
MODEL_DIR = "/opt/airflow/models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_classifier.pkl")

# Paths for storing insights
INSIGHT_DIR = "/opt/airflow/dags/data/insights"
os.makedirs(INSIGHT_DIR, exist_ok=True)

# Inputs
COMBO_COLS = [
    ['road_surface_conditions', 'light_conditions', 'area_accident_occured'],
    ['day_of_week', 'type_of_vehicle', 'weather_conditions']
]

# Table names
TABLE_NAME_CLEANED = "cleaned_rta"
TABLE_NAME_ENCODED = "encoded_rta"

# Utility loaders
def load_model():
    return joblib.load(MODEL_PATH)

def load_cleaned_data():
    """Return the cleaned RTA dataframe (readable categorical columns)."""
    conn = get_postgres_conn()
    df = pd.read_sql(f"SELECT * FROM {TABLE_NAME_CLEANED}", conn)
    conn.close()
    return df

def load_encoded_data():
    conn = get_postgres_conn()
    df_encoded = pd.read_sql(f"SELECT * FROM {TABLE_NAME_ENCODED}", conn)
    conn.close()
    return df_encoded

def load_feature_importances(path=FEATURE_IMP_CSV):
    df = pd.read_csv(path, index_col=0, header=None)
    fi = df.iloc[:, 0]
    fi.index.name = 'feature'
    fi.name = 'importance'
    return fi

# Insight 1: Top Risk Conditions
def top_risk_conditions(feature_importance_csv=FEATURE_IMP_CSV, n=10):
    fi = load_feature_importances(feature_importance_csv)
    top_n = fi.sort_values(ascending=False).head(n)
    top_n.to_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", header=["importance"])
    plt.figure(figsize=(8, 6))
    top_n.sort_values().plot(kind="barh")
    plt.title(f"Top {n} Risk Conditions")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/top_risk_conditions.png")
    plt.close()

# Insight 2: Severity by Time
def severity_by_time():
    df = load_cleaned_data()
    df_grouped = df.groupby(["hour", "accident_severity"]).size().unstack(fill_value=0)
    df_grouped.to_csv(f"{INSIGHT_DIR}/severity_by_hour.csv")
    plt.figure(figsize=(12, 6))
    df_grouped.plot(kind="bar", stacked=True)
    plt.title("Accident Severity by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/severity_by_hour.png")
    plt.close()

# Insight 3: Risky Condition Combinations
def risky_condition_combos(combo_cols=COMBO_COLS, top_k=5):
    df = load_cleaned_data()
    severity_map = {'Slight Injury': 1, 'Serious Injury': 2, 'Fatal injury': 3}
    df['severity_num'] = df['accident_severity'].map(severity_map)
    all_results = []
    for cols in combo_cols:
        grp = (
            df.groupby(cols)['severity_num']
              .mean()
              .reset_index()
              .assign(combo=' & '.join(cols))
        )
        all_results.append(grp)
    results = pd.concat(all_results, ignore_index=True)
    top = results.sort_values('severity_num', ascending=False).head(top_k)
    top.to_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.barh(top['combo'] + ': ' + top[combo_cols[0][0]].astype(str), top['severity_num'])
    plt.gca().invert_yaxis()
    plt.title("Top Risky Condition Combos")
    plt.xlabel("Avg. Severity Score")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/risky_condition_combos.png")
    plt.close()

# Insight 4: Suggested Interventions
def suggest_interventions(feature_importance_csv=FEATURE_IMP_CSV, top_n=5):
    fi = load_feature_importances(feature_importance_csv)
    top_factors = fi.sort_values(ascending=False).head(top_n).index.tolist()
    interventions = []
    for factor in top_factors:
        if "junction" in factor.lower():
            rec = "Redesign junctions with clearer signage or smart lights."
        elif "daylight" in factor.lower():
            rec = "Schedule roadworks after dark to mitigate daylight risks."
        elif "office" in factor.lower():
            rec = "Install speed bumps or traffic patrols in office areas."
        else:
            rec = "Review this condition and consider targeted safety measures."
        interventions.append((factor, rec))
    pd.DataFrame(interventions, columns=["condition", "recommendation"]).to_csv(
        f"{INSIGHT_DIR}/suggested_interventions.csv", index=False)

# Insight 5: Accidents by Road Surface
def accidents_by_road_surface():
    df = load_cleaned_data()
    pivot = df.groupby(["road_surface_conditions", "accident_severity"]).size().unstack(fill_value=0)
    pivot.to_csv(f"{INSIGHT_DIR}/accidents_by_road_surface.csv")
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accidents by Road-Surface Condition & Severity")
    plt.xlabel("Road-surface condition")
    plt.ylabel("Number of accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/accidents_by_road_surface.png")
    plt.close()

# Insight 6: Accidents by Weather
def accidents_by_weather():
    df = load_cleaned_data()
    pivot = df.groupby(["weather_conditions", "accident_severity"]).size().unstack(fill_value=0)
    pivot.to_csv(f"{INSIGHT_DIR}/accidents_by_weather.csv")
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accidents by Weather Condition & Severity")
    plt.xlabel("Weather condition")
    plt.ylabel("Number of accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/accidents_by_weather.png")
    plt.close()

# Insight 7: Heatmap of Road-Surface × Weather
def road_weather_heatmap():
    df = load_cleaned_data()
    heat = df.groupby(["road_surface_conditions", "weather_conditions"]).size().unstack(fill_value=0)
    heat.to_csv(f"{INSIGHT_DIR}/road_weather_heatmap.csv")
    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, cmap="Reds", annot=False)
    plt.title("Accident Count Heatmap: Road-Surface × Weather")
    plt.xlabel("Weather condition")
    plt.ylabel("Road-surface condition")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/road_weather_heatmap.png")
    plt.close()

# Define DAG and tasks
with DAG(
    dag_id='generate_insights_rta_dataset',
    start_date=datetime(2025, 3, 24),
    schedule_interval='@once',
    catchup=False,
    tags=['rta', 'insights']
) as dag:

    top_risk_conditions_task = PythonOperator(
        task_id='top_risk_conditions',
        python_callable=top_risk_conditions,
        op_kwargs={'feature_importance_csv': FEATURE_IMP_CSV, 'n': 10}
    )

    severity_by_time_task = PythonOperator(
        task_id='severity_by_time',
        python_callable=severity_by_time
    )

    accidents_by_road_surface_task = PythonOperator(
        task_id='accidents_by_road_surface',
        python_callable=accidents_by_road_surface
    )

    accidents_by_weather_task = PythonOperator(
        task_id='accidents_by_weather',
        python_callable=accidents_by_weather
    )

    risky_condition_combos_task = PythonOperator(
        task_id='risky_condition_combos',
        python_callable=risky_condition_combos,
        op_kwargs={'combo_cols': COMBO_COLS, 'top_k': 5}
    )

    road_weather_heatmap_task = PythonOperator(
        task_id='road_weather_heatmap',
        python_callable=road_weather_heatmap
    )

    suggest_interventions_task = PythonOperator(
        task_id='suggest_interventions',
        python_callable=suggest_interventions,
        op_kwargs={'feature_importance_csv': FEATURE_IMP_CSV, 'top_n': 5}
    )

    # Set task dependencies
    top_risk_conditions_task >> severity_by_time_task >> accidents_by_road_surface_task \
        >> accidents_by_weather_task >> risky_condition_combos_task \
        >> road_weather_heatmap_task >> suggest_interventions_task

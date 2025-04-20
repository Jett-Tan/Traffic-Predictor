from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from load_rta_dataset import get_postgres_conn

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

# Postgres Connection
POSTGRES_CONN_ID = "postgres_default"
TABLE_NAME_CLEANED = "cleaned_rta"
TABLE_NAME_ENCODED = "encoded_rta"

def load_model():
    return joblib.load(MODEL_PATH)

def load_encoded_data():
    conn = get_postgres_conn()
    df_encoded = pd.read_sql("SELECT * FROM encoded_rta", conn)
    return df_encoded

def load_feature_importances(path=FEATURE_IMP_CSV):
    df = pd.read_csv(path, index_col=0, header=None)
    fi = df.iloc[:, 0]  # Get the first (and only) column as a Series
    
    fi.index.name = 'feature'
    fi.name = 'importance'
    print(fi.head())
    return fi

# Insight 1: Top Risk Conditions
def top_risk_conditions(feature_importance_csv=FEATURE_IMP_CSV, n = 10):
    fi = load_feature_importances(feature_importance_csv)

    # Filter for the top n and write to csv
    top_n = fi.sort_values(ascending=False).head(n)
    top_n.to_csv(f"{INSIGHT_DIR}/top_risk_conditions.csv", header=["importance"])

    # Plot horizontal bar chart of the top risk features
    plt.figure(figsize=(8, 6))
    top_n.sort_values().plot(kind="barh")
    plt.title(f"Top {n} Risk Conditions")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/top_risk_conditions.png")
    plt.close()

# Insight 2: Severity by time
def severity_by_time():
    # Reload from cleaned data (non-encoded) for readable hour/day info
    conn = get_postgres_conn()
    df_cleaned = pd.read_sql("SELECT * FROM cleaned_rta", conn)

    # Group by hour & severity
    df_grouped = df_cleaned.groupby(["hour", "accident_severity"]).size().unstack().fillna(0)
    df_grouped.to_csv(f"{INSIGHT_DIR}/severity_by_hour.csv")

    # Plot stacked bar chart of accidents by severity each hour
    plt.figure(figsize=(12, 6))
    df_grouped.plot(kind="bar", stacked=True)
    plt.title("Accident Severity by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/severity_by_hour.png")
    plt.close()

# Insight 3: Risky Condition Combinations
def risky_condition_combos(combo_cols, top_k=5):
    # Load cleaned data
    conn = get_postgres_conn()
    df = pd.read_sql("SELECT * FROM cleaned_rta", conn)

    # Map to numeric for averaging
    severity_map = {'Slight Injury': 1, 'Serious Injury': 2, 'Fatal injury': 3}
    df['severity_num'] = df['accident_severity'].map(severity_map)

    all_results = []
    for cols in combo_cols:
        grp = (
            df
            .groupby(cols)['severity_num']
            .mean()
            .reset_index()
            .assign(combo=' & '.join(cols))
        )
        all_results.append(grp)
    
    results = pd.concat(all_results, ignore_index=True)
    top = results.sort_values('severity_num', ascending=False).head(top_k)
    top.to_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv", index=False)

    # Simple bar chart on mean severity
    plt.figure(figsize=(10, 6))
    plt.barh(top['combo'] + ': ' + top[cols[0]].astype(str), top['severity_num'])
    plt.gca().invert_yaxis()
    plt.title("Top Risky Condition Combos")
    plt.xlabel("Avg. Severity Score")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/risky_condition_combos.png")
    plt.close()

# Insight 4: Suggested Interventions (TODO: refine recommendations)
def suggest_interventions(feature_importance_csv=FEATURE_IMP_CSV, top_n = 5):
    fi = load_feature_importances(feature_importance_csv)

    top_factors = fi.sort_values(ascending=False).head(top_n).index.tolist()
    interventions = []

    for factor in top_factors:
        if "junction" in factor.lower():
            interventions.append((factor, 
                "Redesign junctions with clearer signage or smart lights."))
        elif "daylight" in factor.lower():
            interventions.append((factor, 
                "Schedule roadworks after dark to mitigate daylight risks."))
        elif "office" in factor.lower():
            interventions.append((factor, 
                "Install speed bumps or traffic patrols in office areas at peak times."))
        else:
            interventions.append((factor, 
                "Review this condition and consider targeted safety measures."))

    pd.DataFrame(interventions, columns=["condition", "recommendation"]).to_csv(f"{INSIGHT_DIR}/suggested_interventions.csv", index=False)


def load_cleaned_data():
    """Return the cleaned RTA dataframe (readable categorical columns)."""
    conn = get_postgres_conn()
    df_cleaned = pd.read_sql(f"SELECT * FROM {TABLE_NAME_CLEANED}", conn)
    conn.close()
    return df_cleaned

# Insight 5: Road‑surface condition vs accident severity
def accidents_by_road_surface():
    df = load_cleaned_data()

    # Count accidents per (surface, severity)
    pivot = (
        df.groupby(["road_surface_conditions", "accident_severity"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    csv_path = f"{INSIGHT_DIR}/accidents_by_road_surface.csv"
    pivot.to_csv(csv_path)

    # Stacked bar chart
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accidents by Road‑Surface Condition & Severity")
    plt.xlabel("Road‑surface condition")
    plt.ylabel("Number of accidents")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/accidents_by_road_surface.png"
    plt.savefig(png_path)

    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()


# Insight 6: Weather condition vs accident severity
def accidents_by_weather():
    df = load_cleaned_data()

    pivot = (
        df.groupby(["weather_conditions", "accident_severity"])
          .size()
          .unstack(fill_value=0)
          .sort_index()
    )
    csv_path = f"{INSIGHT_DIR}/accidents_by_weather.csv"
    pivot.to_csv(csv_path)

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accidents by Weather Condition & Severity")
    plt.xlabel("Weather condition")
    plt.ylabel("Number of accidents")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/accidents_by_weather.png"
    plt.savefig(png_path)

    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()


# Insight 7: Heat‑map of Road‑surface  ×  Weather combos
def road_weather_heatmap():
    df = load_cleaned_data()

    heat = (
        df.groupby(["road_surface_conditions", "weather_conditions"])
          .size()
          .unstack(fill_value=0)
    )
    # Save matrix
    heat.to_csv(f"{INSIGHT_DIR}/road_weather_heatmap.csv")

    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heat, cmap="Reds", annot=False)
    plt.title("Accident count heat‑map:  road surface  ×  weather")
    plt.xlabel("Weather condition")
    plt.ylabel("Road‑surface condition")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/road_weather_heatmap.png")

    plt.close()


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

    risky_condition_combos_task = PythonOperator(
        task_id='risky_condition_combos',
        python_callable=risky_condition_combos,
        op_kwargs={
            'combo_cols': COMBO_COLS,
            'top_k': 5
        }
    )

    suggest_interventions_task = PythonOperator(
        task_id='suggest_interventions',
        python_callable=suggest_interventions,
        op_kwargs={'feature_importance_csv': FEATURE_IMP_CSV, 'top_n': 5}
    )

    accidents_by_road_surface_task = PythonOperator(
        task_id='accidents_by_road_surface',
        python_callable=accidents_by_road_surface,
    )

    accidents_by_weather_task = PythonOperator(
        task_id='accidents_by_weather',
        python_callable=accidents_by_weather,
    )

    road_weather_heatmap_task = PythonOperator(
        task_id='road_weather_heatmap',
        python_callable=road_weather_heatmap,
    )

    

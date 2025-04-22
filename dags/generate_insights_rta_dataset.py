from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
from itertools import combinations


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
    df = pd.read_csv(path, header=0, index_col=False)

    return pd.Series(df["importance"].values, index=df["feature"])

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

     # Group and reshape to long format
    df_grouped = (
        df_cleaned
        .groupby(["hour", "accident_severity"])
        .size()
        .reset_index(name="count")
    )

    # Save in long format for animation use
    df_grouped.to_csv(f"{INSIGHT_DIR}/severity_by_hour.csv", index=False)

    # Plot stacked bar chart of accidents by severity each hour
    pivot_df = df_grouped.pivot(index="hour", columns="accident_severity", values="count").fillna(0)
    plt.figure(figsize=(12, 6))
    pivot_df.plot(kind="bar", stacked=True)
    plt.title("Accident Severity by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/severity_by_hour.png")
    plt.close()

# Insight 3: Risky Condition Combinations
def risky_condition_combos(max_combo_size=3, top_k=5):
    # Load cleaned data
    conn = get_postgres_conn()
    df = pd.read_sql("SELECT * FROM cleaned_rta", conn)

    # Map to numeric for averaging
    severity_map = {'Slight Injury': 1, 'Serious Injury': 2, 'Fatal injury': 3}
    df['severity_num'] = df['accident_severity'].map(severity_map)

    # Top raw categorical features from our model insights
    top_categorical_features = [
        "light_conditions", "vehicle_movement", "types_of_junction",
        "age_band_of_driver", "educational_level", "lanes_or_medians",
        "area_accident_occured", "type_of_collision", "vehicle_driver_relation"
    ]

    all_results = []
    for combo in combinations(top_categorical_features, max_combo_size):
        try:
            group = (
                df
                .groupby(list(combo))
                .agg(severity_num=('severity_num', 'mean'), support=('severity_num', 'count'))
                .reset_index()
            )

            group["combo_name"] = (
                group[list(combo)].astype(str).agg(" & ".join, axis=1)
            )
            group["combo"] = " + ".join(combo)
            all_results.append(group[["combo_name", "severity_num", "support", "combo"]])
        except KeyError as e:
            print(f"Skipped combo {combo} due to missing column: {e}")

    if not all_results:
        print("⚠️ No valid combos generated.")
        return

    # Combine all results and sort
    results = pd.concat(all_results, ignore_index=True)
    top = results.sort_values(["severity_num", "support"], ascending=[False, False]).head(top_k)

    # Save to CSV
    top.to_csv(f"{INSIGHT_DIR}/risky_condition_combos.csv", index=False)

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.barh(top['combo_name'], top['severity_num'], color='crimson')
    plt.gca().invert_yaxis()
    plt.title("Top Risky Condition Combos (Severity & Frequency)")
    plt.xlabel("Average Severity Score")

    # Add support count as annotation
    for bar, support in zip(bars, top['support']):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{support} cases", va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/risky_condition_combos.png")
    plt.close()

# Insight 4: Uses the feature importance CSV to derive top features + rule-based recommendations
# Refer to dashboard code

def load_cleaned_data():
    """Return the cleaned RTA dataframe (readable categorical columns)."""
    conn = get_postgres_conn()
    df_cleaned = pd.read_sql(f"SELECT * FROM {TABLE_NAME_CLEANED}", conn)
    conn.close()
    return df_cleaned

# Insight 5: Driver Age Band vs Accident Severity
def age_band_vs_severity():
    df = load_cleaned_data()
    pivot = df.groupby(["age_band_of_driver", "accident_severity"]).size().unstack(fill_value=0).sort_index()
    csv_path = f"{INSIGHT_DIR}/age_band_vs_severity.csv"
    pivot.to_csv(csv_path)
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accident Severity by Driver Age Band")
    plt.xlabel("Age Band of Driver")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/age_band_vs_severity.png"
    plt.savefig(png_path)
    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()

# Insight 6: Lanes or Medians vs Vehicles Involved 
def lanes_vs_vehicles():
    df = load_cleaned_data()
    lanes_summary = df['lanes_or_medians'].value_counts().reset_index()
    lanes_summary.columns = ['Lane Type', 'Number of Accidents']
    lanes_summary.to_csv(f"{INSIGHT_DIR}/accidents_by_lane.csv", index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=lanes_summary, x='Lane Type', y='Number of Accidents')
    plt.title("Accidents by Lane Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/accidents_by_lane.png")
    plt.close()

# Insight 7: Cause of Accident
def accidents_by_cause():
    df = load_cleaned_data()

    # Count number of accidents per cause
    cause_counts = df["cause_of_accident"].value_counts()

    # Save CSV
    csv_path = f"{INSIGHT_DIR}/accidents_by_cause.csv"
    cause_counts.to_csv(csv_path, header=["count"])

    # Plot setup
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    colors = plt.get_cmap("Set2").colors

    # Horizontal bar chart (good for long cause names)
    cause_counts.sort_values().plot(kind="barh", color=colors[0])
    plt.title("Number of Accidents by Cause", fontsize=16)
    plt.xlabel("Number of Accidents", fontsize=12)
    plt.ylabel("Cause of Accident", fontsize=12)
    plt.tight_layout()

    # Save
    png_path = f"{INSIGHT_DIR}/accidents_by_cause.png"
    plt.savefig(png_path)
    print(f"Saved visual to: {png_path}")
    plt.close()

# Insight 8: Driver Experience vs Severity of Accident
def driver_experience_vs_severity():
    df = load_cleaned_data()
    pivot = df.groupby(["driving_experience", "accident_severity"]).size().unstack(fill_value=0)
    pivot.to_csv(f"{INSIGHT_DIR}/driver_experience_vs_severity.csv")

    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Accident Severity by Driver Experience")
    plt.xlabel("Driving Experience")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/driver_experience_vs_severity.png")
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
            "max_combo_size": 3,
            'top_k': 5
        }
    )
    
    age_band_vs_severity_task = PythonOperator(
    task_id='age_band_vs_severity',
    python_callable=age_band_vs_severity,
    )

    lanes_vs_vehicles_task = PythonOperator(
        task_id='lanes_vs_vehicles',
        python_callable=lanes_vs_vehicles,
    )

    accidents_by_cause_task = PythonOperator(
    task_id='accidents_by_cause',
    python_callable = accidents_by_cause,
    )

    experience_vs_severity_task = PythonOperator(
        task_id='experience_vs_severity',
        python_callable=driver_experience_vs_severity,
    )



    

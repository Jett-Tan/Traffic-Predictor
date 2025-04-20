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
TABLE_NAME_CLEANED = "rta"
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
    df_cleaned = pd.read_sql("SELECT * FROM rta", conn)

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

# Insight 8: Driver Age Band vs Accident Severity
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

# Insight 9: Time of Day vs Collision Type
def collision_type_by_hour():
    df = load_cleaned_data()
    pivot = df.groupby(["hour", "type_of_collision"]).size().unstack(fill_value=0).sort_index()
    csv_path = f"{INSIGHT_DIR}/collision_type_by_hour.csv"
    pivot.to_csv(csv_path)
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Collision Types by Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Number of Collisions")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/collision_type_by_hour.png"
    plt.savefig(png_path)
    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()

# Insight 10: Lanes or Medians vs Vehicles Involved
# Insight: Lanes or Medians vs Vehicles Involved (refined)
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




# Insight 11: Cause of Accident
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


# Insight 12: Driver-Casualty Relation vs Severity
def relation_vs_casualty_severity():
    df = load_cleaned_data()
    pivot = df.groupby(["vehicle_driver_relation", "casualty_severity"]).size().unstack(fill_value=0).sort_index()
    csv_path = f"{INSIGHT_DIR}/relation_vs_casualty_severity.csv"
    pivot.to_csv(csv_path)
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Casualty Severity by Driver-Casualty Relation")
    plt.xlabel("Vehicle-Driver Relation")
    plt.ylabel("Number of Casualties")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/relation_vs_casualty_severity.png"
    plt.savefig(png_path)
    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()

# Insight 13: Educational Level vs Accident Cause
def education_vs_cause():
    df = load_cleaned_data()
    pivot = df.groupby(["educational_level", "cause_of_accident"]).size().unstack(fill_value=0).sort_index()
    csv_path = f"{INSIGHT_DIR}/education_vs_cause.csv"
    pivot.to_csv(csv_path)
    plt.figure(figsize=(10, 6))
    pivot.plot(kind="bar", stacked=True)
    plt.title("Cause of Accident by Educational Level")
    plt.xlabel("Educational Level")
    plt.ylabel("Number of Accidents")
    plt.tight_layout()
    png_path = f"{INSIGHT_DIR}/education_vs_cause.png"
    plt.savefig(png_path)
    print(f"Saved image to: {png_path}")
    print(f"Exists: {os.path.exists(png_path)}")
    plt.close()


# Insight 15 defect type vs accident count
def vehicle_defect_vs_accidents():
    conn = get_postgres_conn()
    df = pd.read_sql("SELECT * FROM rta", conn)

    defect_summary = df['defect_of_vehicle'].value_counts().reset_index()
    defect_summary.columns = ['Vehicle Defect', 'Accident Count']

    # Save CSV
    defect_summary.to_csv(f"{INSIGHT_DIR}/vehicle_defect_vs_accidents.csv", index=False)

    # Save plot
    plt.figure(figsize=(10, 6))
    plt.bar(defect_summary['Vehicle Defect'], defect_summary['Accident Count'])
    plt.xticks(rotation=45, ha='right')
    plt.title("Vehicle Defect Type vs Accident Count")
    plt.tight_layout()
    plt.savefig(f"{INSIGHT_DIR}/vehicle_defect_vs_accidents.png")
    plt.close()



# Insight 16: Service Year of Vehicle vs Accident Frequency
def service_year_vs_accidents():
    df = load_cleaned_data()
    
    year_counts = df["service_year_of_vehicle"].value_counts().sort_index()
    csv_path = f"{INSIGHT_DIR}/service_year_vs_accidents.csv"
    year_counts.to_csv(csv_path, header=["accident_count"])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=year_counts.index, y=year_counts.values)
    plt.title("Accident Frequency by Service Year of Vehicle")
    plt.xlabel("Service Year")
    plt.ylabel("Number of Accidents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    png_path = f"{INSIGHT_DIR}/service_year_vs_accidents.png"
    plt.savefig(png_path)
    plt.close()

# Insight 17: driver experience and severity of accident
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

    age_band_vs_severity_task = PythonOperator(
    task_id='age_band_vs_severity',
    python_callable=age_band_vs_severity,
    )

    collision_type_by_hour_task = PythonOperator(
        task_id='collision_type_by_hour',
        python_callable=collision_type_by_hour,
    )

    lanes_vs_vehicles_task = PythonOperator(
        task_id='lanes_vs_vehicles',
        python_callable=lanes_vs_vehicles,
    )

    accidents_by_cause_task = PythonOperator(
    task_id='accidents_by_cause',
    python_callable = accidents_by_cause,
    )

    driver_experience_vs_severity_task = PythonOperator(
    task_id='driver_experience_vs_severity',
    python_callable=driver_experience_vs_severity,
    )

   
    relation_vs_casualty_severity_task = PythonOperator(
        task_id='relation_vs_casualty_severity',
        python_callable=relation_vs_casualty_severity,
    )

    education_vs_cause_task = PythonOperator(
        task_id='education_vs_cause',
        python_callable=education_vs_cause,
    )

    vehicle_defect_vs_accidents_task = PythonOperator(
        task_id="vehicle_defect_vs_accidents",
        python_callable=vehicle_defect_vs_accidents,
    )


    service_year_vs_accidents_task = PythonOperator(
        task_id='service_year_vs_accidents',
        python_callable=service_year_vs_accidents,
    )

    experience_vs_severity_task = PythonOperator(
        task_id='experience_vs_severity',
        python_callable=driver_experience_vs_severity,
    )

top_risk_conditions_task >> [
    severity_by_time_task,
    risky_condition_combos_task,
    suggest_interventions_task,
    accidents_by_road_surface_task,
    accidents_by_weather_task,
    road_weather_heatmap_task,
    age_band_vs_severity_task,
    collision_type_by_hour_task,
    lanes_vs_vehicles_task,
    accidents_by_cause_task,
    relation_vs_casualty_severity_task,
    education_vs_cause_task,
    driver_experience_vs_severity_task,
    vehicle_defect_vs_accidents_task,
    service_year_vs_accidents_task,
    experience_vs_severity_task
]

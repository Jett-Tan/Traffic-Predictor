from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from geopy.distance import geodesic
import re

import logging
log = logging.getLogger("airflow.task")

DATA_PATH = "/opt/airflow/dags/data/RTA_Dataset.csv"  # Adjust this
MODEL_PATH = "/opt/airflow/dags/data/accident_severity_model.pkl"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    "retries": 1
}

with DAG(
    dag_id="train_accident_model_dag",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["ml", "training","IS3107"]
) as dag:

    def load_and_preprocess():
        df = pd.read_csv(DATA_PATH)

        # Drop unnecessary columns
        df.drop(columns=[
            'Defect_of_vehicle', 'Service_year_of_vehicle', 
            'Work_of_casuality', 'Fitness_of_casuality'
        ], inplace=True, errors='ignore')

        # Clean column names
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        df.rename(columns={"road_allignment": "road_alignment"}, inplace=True)

        # Convert time to hour
        if 'time' in df.columns:
            df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S", errors='coerce')
            df["hour"] = df["time"].dt.hour
        else:
            df["hour"] = 0  # fallback in case 'time' missing

        # Select final columns
        selected_features = ["area_accident_occured", "lanes_or_medians", "types_of_junction", "weather_conditions", "hour"]
        target = "accident_severity"
        df = df[selected_features + [target]].dropna()

        # transform categorical features
        types_of_junction_to_types_of_junction = {
            "Y Shape":"y_shape",
            "No junction":"no_junction",
            "Crossing":  "crossing",
            "Other":  "other",
            "Unknown":  "unknown",
            "O Shape":"o_shape",
            "T Shape":"t_shape",
            "X Shape":"x_shape",
        }
        df["types_of_junction"] = df["types_of_junction"].map(types_of_junction_to_types_of_junction)

        area_to_highway = {
            "Other": "road",
            "Office areas": "service",
            "Residential areas": "residential",
            "Church areas": "service",
            "Industrial areas": "service",
            "School areas": "living_street",
            "Recreational areas": "living_street",
            "Outside rural areas": "unclassified",
            "Hospital areas": "service",
            "Market areas": "living_street",
            "Rural village areas": "tertiary",
            "Unknown": "road",
            "Rural village areasOffice areas": "road",  # Inconsistent value
            # Possible trimmed version to cover any leading/trailing whitespace
            "  Recreational areas": "living_street",
            "  Market areas": "living_street"
        }
        df["area_accident_occured"] = df["area_accident_occured"].map(area_to_highway)
        df["lanes_or_medians"] = df["lanes_or_medians"].apply(
            lambda x: "two_way" if 'two way' in x.lower() or 'two-way' in x.lower()
            else "one_way" if 'double carriageway' in x.lower()
            else x.strip().replace(" ", "_").lower()
        )
        df["weather_conditions"] = df["weather_conditions"].apply(
            lambda x: "rain" if "rain" in x.lower() else "no_rain"
        )
        # Encode all categorical values
        encoders = {}
        for col in selected_features + [target]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            log.info(encoders[col].classes_)
        # Save encoders for future use
        with open("/opt/airflow/dags/tmp/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)

        df.to_csv("/opt/airflow/dags/tmp/processed_rta.csv", index=False)

    def train_model():
        df = pd.read_csv("/opt/airflow/dags/tmp/processed_rta.csv")
        X = df.drop("accident_severity", axis=1)
        y = df["accident_severity"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

    def extract_hour(message):
        # Extract hour from message using regex
        log.info(f"Extracting hour from message: {message}")
        match = re.search(r"\)(\d{1,2}):(\d{2})", message)
        return int(match.group(1)) if match else 12

    def infer_area_from_message(message):
        log.info(f"Inferring area from message: {message}")
        return "road"
        # roads = ["AYE", "PIE", "SLE", "CTE", "TPE", "ECP"]
        # for road in roads:
        #     if road in message:
        #         return road
        # return "Other"

    def get_nearest_lane_type(row, roads_df):
        log.info(f"Finding nearest lane type for coordinates: {row['latitude']}, {row['longitude']}")
        return "one_way"  # Placeholder for actual logic
        # point = (row["latitude"], row["longitude"])
        # roads_df["distance"] = roads_df.apply(
        #     lambda r: min(
        #         geodesic(point, (r["start_lat"], r["start_lon"])).meters,
        #         geodesic(point, (r["end_lat"], r["end_lon"])).meters
        #     ), axis=1
        # )
        # nearest = roads_df.loc[roads_df["distance"].idxmin()]
        # return "two_way" if nearest["two_way"] else "one_way"

    def get_nearest_junction_type(row, junction_df, max_dist=0.0005):
        log.info(f"Finding nearest junction type for coordinates: {row['latitude']}, {row['longitude']}")
        return "no_junction"  # Placeholder for actual logic
        # for _, j in junction_df.iterrows():
        #     dist = ((row["latitude"] - j["latitude"])**2 + (row["longitude"] - j["longitude"])**2)**0.5
        #     if dist < max_dist:
        #         return j["junction_type"]
        # return "no_junction"
    
    def predict_batch():
      # Paths
      ENCODERS_PATH = "/opt/airflow/dags/tmp/encoders.pkl"
      OUTPUT_PATH = "/opt/airflow/dags/tmp/predictions.csv"
      POSTGRES_DATA_PATH = "/opt/airflow/dags/data/live_traffic/live_traffic_incidents_postgres.csv"  # Adjust this
      CSV_ROADS = "/opt/airflow/dags/data/road_metadata.csv"
      CSV_JUNCTIONS = "/opt/airflow/dags/data/junctions_from_geojson.csv"
      # --- Load data ---
      log.info("Reading input files...")
      df = pd.read_csv(POSTGRES_DATA_PATH)
      roads_df = pd.read_csv(CSV_ROADS)
      junction_df = pd.read_csv(CSV_JUNCTIONS)

      # --- Filter only accidents ---
      log.info("Filtering accidents...")
      df = df[df["type"].str.lower() == "accident"].copy()
      roads_df = roads_df.dropna(subset=["road_name"])

      # --- Feature Engineering ---
      log.info("Feature engineering...")
      df["hour"] = df["message"].apply(extract_hour)
      df["area_accident_occured"] = df["message"].apply(infer_area_from_message)
      df["lanes_or_medians"] = df.apply(lambda row: get_nearest_lane_type(row, roads_df), axis=1)
      df["types_of_junction"] = df.apply(lambda row: get_nearest_junction_type(row, junction_df), axis=1)
      df["weather_conditions"] = "no_rain" 
      
      with open(MODEL_PATH, "rb") as f:
          model = pickle.load(f)
      with open(ENCODERS_PATH, "rb") as f:
          encoders = pickle.load(f)

      # Preprocess input
      log.info("Preprocessing input data...")
      features = ["area_accident_occured", "lanes_or_medians", "types_of_junction", "weather_conditions", "hour"]
      df = df[features].dropna()
      for col in features:
          df[col] = encoders[col].transform(df[col])

      # Predict
      log.info("Predicting accident severity...")
      predictions = model.predict(df)
      predicted_labels = encoders["accident_severity"].inverse_transform(predictions)

      # Save results
      log.info("Saving predictions...")
      df["predicted_accident_severity"] = predicted_labels
      df.to_csv(OUTPUT_PATH, index=False)
      print(f"Saved predictions to {OUTPUT_PATH}")

    preprocess = PythonOperator(
        task_id="load_and_preprocess",
        python_callable=load_and_preprocess
    )
    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )
    run_model = PythonOperator(
        task_id="run_model",
        python_callable=predict_batch,
        execution_timeout=timedelta(minutes=5),
        retries=1,
        retry_delay=timedelta(minutes=1),
    )

    preprocess >> train >> run_model

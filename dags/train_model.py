import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve,classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import math
import dabl
import json
from logging import Logger
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import LineString, Point


# Train using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

def train_model():
  file_path = "dags/data/RTA_Dataset.csv"
  df = pd.read_csv(file_path)
  # Remove unnecessary columns

  # Decision: Remove Defect_of_vehicle, Service_year_of_vehicle, Work_of_casuality, Fitness_of_casuality
  # Because these attributes are unrelated in our prediction of traffic accidents based on traffic conditions and high percentage of null values
  df_cleaned = df.drop(columns=['Defect_of_vehicle', 'Service_year_of_vehicle', 'Work_of_casuality', 'Fitness_of_casuality'])

  # Standardize column values (lowercase, strip spaces, replace underscores)
  df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(" ", "_")

  # Rename the spelling error
  df_cleaned.rename(columns={"road_allignment": "road_alignment"}, inplace=True)

  # Time column conversion
  # Convert time to datetime format and extract hour
  df_cleaned["time"] = pd.to_datetime(df_cleaned["time"], format="%H:%M:%S")
  df_cleaned["hour"] = df_cleaned["time"].dt.hour # group by the hour

  # There are many NA values in casualty_class and sex_of_casualty. 
  # Since these are not directly related to the goal of our prediction, we can drop these columns 
  df_cleaned.drop(columns=["casualty_class", "sex_of_casualty"], inplace=True)

  # Fill missing categorical values with mode
  categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
  for col in categorical_cols:
      df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

  # Further drop irrelevant columns
  df_cleaned.drop(columns=["pedestrian_movement", "road_surface_type", "time"], inplace=True)

  # Strip whitespace from all string columns
  df_cleaned = df_cleaned.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

  categorical_cols = df_cleaned.select_dtypes(include="object").columns  

  df_cleaned[categorical_cols].nunique().sort_values(ascending=False)

  # Encode the target variable accident_severity
  print(df_cleaned['accident_severity'].unique())
  df_cleaned['accident_severity'] = df_cleaned['accident_severity'].map({
      'Slight Injury': 0, 'Serious Injury': 1, 'Fatal injury': 2
  })

  # One-hot encode categorical features
  filtered_cols = [col for col in categorical_cols if col not in ['accident_severity']]
  df_encoded = pd.get_dummies(df_cleaned[filtered_cols]) 
  # Ensure all features are numeric (convert bools to ints)
  df_encoded = df_encoded.astype(int)

  # Encode the features accident_severity
  print(df_cleaned['types_of_junction'].unique())
  df_cleaned['accident_severity'] = df_cleaned['accident_severity'].map({
      'Slight Injury': 0, 'Serious Injury': 1, 'Fatal injury': 2
  })

  # One-hot encode categorical features
  filtered_cols = [col for col in categorical_cols if col not in ['accident_severity']]
  df_encoded = pd.get_dummies(df_cleaned[filtered_cols]) 
  # Ensure all features are numeric (convert bools to ints)
  df_encoded = df_encoded.astype(int)

  # Define features and target
  X = df_encoded
  y = df_cleaned['accident_severity']


  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

  # Imbalance data treatment using SMOTE only to training data
  smote = SMOTE(random_state=42)
  X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

  # New class distribution
  print("Before SMOTE:", Counter(y_train))
  print("After SMOTE: ", Counter(y_train_resampled))

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train_resampled, y_train_resampled)

  # Evaluate Model Performance

  y_pred = model.predict(X_test)


with DAG("get_road_type_dag",
         default_args=default_args,
         schedule_interval="@daily",
         start_date=datetime(2025, 1, 1),
         catchup=False,
         tags=["traffic", "geocoding"]) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    train_model 
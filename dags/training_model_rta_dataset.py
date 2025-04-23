from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import time

from load_rta_dataset import get_postgres_conn

# Paths
CSV_FILE_PATH_CLEANED = "/opt/airflow/dags/data/RTA_Dataset_Cleaned.csv"
CSV_FILE_PATH_ENCODED = "/opt/airflow/dags/data/RTA_Dataset_Encoded.csv"

# Paths for storing models
MODEL_DIR = "/opt/airflow/models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_classifier.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

# Postgres Connection
POSTGRES_CONN_ID = "postgres_default"
TABLE_NAME_CLEANED = "cleaned_rta"
TABLE_NAME_ENCODED = "encoded_rta"

def encode_features():
    conn = get_postgres_conn()
    cursor = conn.cursor()

    df_cleaned = pd.read_sql(f"SELECT * FROM {TABLE_NAME_CLEANED}", conn)

    categorical_cols = df_cleaned.select_dtypes(include="object").columns  
    # Encode the target variable accident_severity
    df_cleaned['accident_severity'] = df_cleaned['accident_severity'].map({
    'Slight Injury': 0, 'Serious Injury': 1, 'Fatal injury': 2
    })

    # One-hot encode categorical features
    filtered_cols = [col for col in categorical_cols if col not in ['accident_severity']]
    df_encoded = pd.get_dummies(df_cleaned[filtered_cols]) 
    # Ensure no syntax error from column names not accepted by Postgres
    df_encoded.columns = df_encoded.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    # Ensure all features are numeric (convert bools to ints)
    df_encoded = df_encoded.astype(int)

    # Join the encoded features and the target variable (accident_severity)
    df_final = pd.concat([df_encoded, df_cleaned['accident_severity']], axis=1)
    df_final.to_csv(CSV_FILE_PATH_ENCODED, index=False)

def create_encoded_RTA_table():
    # Database connection
    conn = get_postgres_conn()
    cursor = conn.cursor()

    # Load the CSV once just to infer the columns
    df = pd.read_csv(CSV_FILE_PATH_ENCODED)
    
    # Dynamically build the CREATE TABLE query
    column_defs = []
    for col in df.columns:
        if col == "accident_severity":
            column_defs.append(f"{col} INT")
        else:
            column_defs.append(f"{col} INT")
    
    create_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME_ENCODED} (
            id SERIAL PRIMARY KEY,
            {', '.join(column_defs)}
        );
    """
    
    cursor.execute(create_query)
    conn.commit()
    cursor.close()
    conn.close()

def drop_encoded_RTA_table():
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        DROP TABLE IF EXISTS {TABLE_NAME_ENCODED};
    """)
    conn.commit()
    cursor.close()
    conn.close()

def insert_encoded_data():
    # Read CSV
    df = pd.read_csv(CSV_FILE_PATH_ENCODED)
    cols = list(df.columns)
    
    conn = get_postgres_conn()
    cursor = conn.cursor()

    # Insert data
    for _, row in df.iterrows():
        values = tuple(row)
        placeholders = ','.join(['%s'] * len(values))
        insert_query = f"""
            INSERT INTO {TABLE_NAME_ENCODED} ({', '.join(cols)})
            VALUES ({placeholders})
        """
        cursor.execute(insert_query, values)

    conn.commit()
    cursor.close()
    conn.close()

def training_model():
    conn = get_postgres_conn()
    df_encoded = pd.read_sql(f"SELECT * FROM {TABLE_NAME_ENCODED}", conn)

    # Define features and target
    X = df_encoded.drop(columns=['accident_severity', 'id'])
    y = df_encoded['accident_severity']
    print(X.shape)
    print(y.shape)
    print(y.isna().sum())

   # Train using Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Imbalance data treatment using SMOTE only to training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # New class distribution
    from collections import Counter
    print("Before SMOTE:", Counter(y_train))
    print("After SMOTE: ", Counter(y_train_resampled))

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    start_time = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_duration = time.time() - start_time
    print(f"Training Time: {training_duration:.2f} seconds")

    # Evaluate Model Performance
    from sklearn.metrics import classification_report, confusion_matrix

    start_time = time.time()
    y_pred = model.predict(X_test)  

    inference_duration = time.time() - start_time
    print(f"Inference Time (on {len(X_test)} samples): {inference_duration:.4f} seconds")
    print(f"Average Inference Time per sample: {inference_duration / len(X_test):.6f} seconds")

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at: {MODEL_PATH}")

    # Save feature importance
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.rename("importance").reset_index()
    feature_importances.columns = ["feature", "importance"]
    feature_importances.sort_values(by="importance", ascending=False).to_csv("/opt/airflow/dags/data/feature_importance.csv", index=False)

with DAG(
    dag_id='training_model_rta_dataset',
    start_date=datetime(2025, 3, 24),
    schedule_interval='@once',
    catchup=False,
    tags=['rta', 'encoding_features', 'model_training']
) as dag:
    create_encoded_table_task = PythonOperator(
        task_id='create_encoded_RTA_table',
        python_callable=create_encoded_RTA_table,
    )

    drop_encoded_table_task = PythonOperator(
        task_id='drop_encoded_RTA_table',
        python_callable=drop_encoded_RTA_table,
    )

    insert_encoded_data_task = PythonOperator(
        task_id='insert_encoded_data',
        python_callable=insert_encoded_data,
    )

    training_model_task = PythonOperator(
        task_id='training_model',
        python_callable=training_model,
    )

    encode_features_task = PythonOperator(
        task_id='encode_features',
        python_callable=encode_features,
    )

    # Task dependencies
    drop_encoded_table_task >> encode_features_task >> create_encoded_table_task >> insert_encoded_data_task >> training_model_task

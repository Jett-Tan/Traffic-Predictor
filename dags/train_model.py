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
import joblib
from imblearn.ensemble import BalancedRandomForestClassifier
import psycopg2

# Train using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}

def train_model1():
        
    # Load the dataset# Load the dataset
    file_path = "/opt/airflow/dags/data/RTA_Dataset.csv"
    df = pd.read_csv(file_path) 

    # Drop all rows with missing values
    df = df.dropna()

    # Select relevant columns
    columns = [
        'day_of_week', 'age_band_of_driver', 'type_of_vehicle',
        'area_accident_occured', 'lanes_or_medians',
        'types_of_junction', 'weather_conditions', 'accident_severity'
    ]

    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    df = df[columns].copy()

    # 1. Clean 'day_of_week'
    valid_days = ['monday', 'sunday', 'friday', 'wednesday', 'saturday', 'thursday', 'tuesday']
    df['day_of_week'] = df['day_of_week'].str.lower()
    df = df[df['day_of_week'].isin(valid_days)]

    # 2. Clean 'age_band_of_driver'
    df = df[df['age_band_of_driver'] != 'Under 18']
    df['age_band_of_driver'] = df['age_band_of_driver'].replace({
        'Over 51': '>51',
        'Unknown': 'unknown'
    })

    # 3. Simplify 'type_of_vehicle'
    car_types = ['Automobile', 'Taxi', 'Stationwagen']
    lorry_types = ['Lorry (41?100Q)', 'Lorry (11?40Q)', 'Long lorry', 'Pick up upto 10Q']
    bus_types = ['Public (> 45 seats)', 'Public (12 seats)', 'Public (13?45 seats)']
    motorcycle_types = ['Motorcycle', 'Bajaj', 'Motorcycle (below 400cc)']
    other_types =['Ridden horse','Other','Special vehicle','Turbo','Bicycle']

    def simplify_vehicle_type(v):
        if v in car_types:
            return 'car'
        elif v in lorry_types:
            return 'lorry'
        elif v in bus_types:
            return 'bus'
        elif v in motorcycle_types:
            return 'motorcycle'
        elif v in other_types:
            return 'other'
        else:
            return 'other'
    df['type_of_vehicle'] = df['type_of_vehicle'].apply(simplify_vehicle_type)

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
    # transform 'area_accident_occured'
    area_to_highway = {
        "Other": "unknown",
        "Office areas": "service",
        "Residential areas": "residential",
        "Church areas": "service",
        "Industrial areas": "service",
        "School areas": "living_street",
        "Recreational areas": "living_street",
        "Outside rural areas": "unknown",
        "Hospital areas": "service",
        "Market areas": "living_street",
        "Rural village areas": "living_street",
        "Unknown": "unknown",
        "Rural village areasOffice areas": "unknown",  # Inconsistent value
        # Possible trimmed version to cover any leading/trailing whitespace
        "  Recreational areas": "living_street",
        "  Market areas": "living_street"
    }
    df["area_accident_occured"] = df["area_accident_occured"].dropna()
    df["area_accident_occured"] = df["area_accident_occured"].map(area_to_highway)

    # transform 'lanes_or_medians'
    df["lanes_or_medians"] = df["lanes_or_medians"].apply(
        lambda x: "two_way" if 'two way' in x.lower() or 'two-way' in x.lower()
        else "one_way" if 'double carriageway' in x.lower()
        else x.strip().replace(" ", "_").lower() 
    )
    # transform 'weather_conditions'
    df["weather_conditions"] = df["weather_conditions"].apply(
        lambda x: "rain" if "rain" in x.lower() else "no_rain"
    )
    
    # transfrom 'accident_severity' # target
    df['accident_severity'] = df['accident_severity'].str.lower()
    df['accident_severity'] = df['accident_severity'].map(lambda x: x.split(' ')[0] if ' ' in x else x)
    df.describe(include='all')
    df.info()
    for col in columns:
        print(df[col].unique())  # Check unique values before encoding
    categorical_cols = df.select_dtypes(include="object").columns  
    # Encode features and target
    df['accident_severity'] = df['accident_severity'].map({
        'slight': 0, 'serious': 1, 'fatal': 2
    })
    # One-hot encode categorical features
    filtered_cols = [col for col in categorical_cols if col not in ['accident_severity']]
    df_encoded = pd.get_dummies(df[filtered_cols]) 
    # Ensure all features are numeric (convert bools to ints)
    df_encoded = df_encoded.astype(int)

    # Define features and target
    X = df_encoded
    y = df['accident_severity']

    print(X.shape)
    print(y.shape)
    print(y.isna().sum())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to balance the training set
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_resampled, y_resampled)
    # Initialize Balanced Random Forest
    brf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model
    brf_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = brf_model.predict(X_test)

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # Feature importance in influencing accident severity

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False)

    print(top_features)

    # Save the top features to a CSV file
    top_features_df = pd.DataFrame({'Feature': top_features.index, 'Importance': top_features.values})
    csv_filename = '/opt/airflow/dags/data/top_feature_importances.csv'
    top_features_df.to_csv(csv_filename, index=False)
    print(f"Top feature importances saved to '{csv_filename}' in the same directory as this notebook.")

def get_postgres_conn():
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )
    return conn

def train_model2():
    # Load and clean dataset
    conn = get_postgres_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM rta;
    """)
    rows = cursor.fetchall()
    # Get column names from cursor
    colnames = [desc[0] for desc in cursor.description]

    # Load into Pandas DataFrame
    df = pd.DataFrame(rows, columns=colnames)
    
    # file_path = "/opt/airflow/dags/data/RTA_Dataset.csv"
    # df = pd.read_csv(file_path)

    # Select relevant columns
    columns = [
        "hour",'day_of_week', 'age_band_of_driver', 'type_of_vehicle',
        'area_accident_occured', 'lanes_or_medians',
        'types_of_junction', 'weather_conditions','vehicle_driver_relation','accident_severity'
    ]

    # Convert all column names to lowercase
    df.columns = df.columns.str.lower()
    # 1 Clean 'time'
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["hour"] = df["time"].dt.hour # group by the hour
    df["hour"] = df["hour"].astype(str).str.zfill(2)  # Ensure two digits for hour
    df = df[columns].copy()


    # 2. Clean 'day_of_week'
    valid_days = ['monday', 'sunday', 'friday', 'wednesday', 'saturday', 'thursday', 'tuesday']
    df['day_of_week'] = df['day_of_week'].str.lower()
    df = df[df['day_of_week'].isin(valid_days)]

    # 3. Clean 'age_band_of_driver'
    df = df[df['age_band_of_driver'] != 'Under 18']
    df['age_band_of_driver'] = df['age_band_of_driver'].replace({
        'Over 51': '>51', 'Unknown': 'unknown'
    })

    # 4. Simplify 'type_of_vehicle'
    def simplify_vehicle_type(v):
        car_types = ['Automobile', 'Taxi', 'Stationwagen']
        lorry_types = ['Lorry (41?100Q)', 'Lorry (11?40Q)', 'Long lorry', 'Pick up upto 10Q']
        bus_types = ['Public (> 45 seats)', 'Public (12 seats)', 'Public (13?45 seats)']
        motorcycle_types = ['Motorcycle', 'Bajaj', 'Motorcycle (below 400cc)']
        other_types = ['Ridden horse', 'Other', 'Special vehicle', 'Turbo', 'Bicycle']
        if v in car_types:
            return 'car'
        elif v in lorry_types:
            return 'lorry'
        elif v in bus_types:
            return 'bus'
        elif v in motorcycle_types:
            return 'motorcycle'
        elif v in other_types:
            return 'other'
        else:
            return 'other'
    df['type_of_vehicle'] = df['type_of_vehicle'].apply(simplify_vehicle_type)

    # 5. Transform 'types_of_junction'
    junction_map = {
        "Y Shape": "y_shape", "No junction": "no_junction", "Crossing": "crossing",
        "Other": "other", "Unknown": "unknown", "O Shape": "o_shape",
        "T Shape": "t_shape", "X Shape": "x_shape"
    }
    df["types_of_junction"] = df["types_of_junction"].map(junction_map)

    # 6. Transform 'area_accident_occured'
    area_map = {
        "Other": "unknown", 
        "Office areas": "service", 
        "Residential areas": "residential",
        "Church areas": "service", 
        "Industrial areas": "service", 
        "School areas": "living_street",
        "Recreational areas": "living_street", 
        "Outside rural areas": "unknown",
        "Hospital areas": "service", 
        "Market areas": "living_street",
        "Rural village areas": "living_street", 
        "Unknown": "unknown",
        "Rural village areasOffice areas": "unknown", 
        "  Recreational areas": "living_street",
        "  Market areas": "living_street"
    }
    df["area_accident_occured"] = df["area_accident_occured"].map(area_map)

    # 7. transform 'lanes_or_medians'
    def clean_lanes(x):
        x = str(x).lower().strip()
        if 'two way' in x or 'two-way' in x:
            return 'two_way'
        elif 'double carriageway' in x:
            return 'one_way'
        elif x == '' or x == 'nan' or x == 'unknown':
            return 'unknown'
        else:
            return x.replace(" ", "_")
    df["lanes_or_medians"] = df["lanes_or_medians"].apply(clean_lanes)

    # 8. transform 'weather_conditions'
    df["weather_conditions"] = df["weather_conditions"].apply(
        lambda x: "rain" if "rain" in x.lower() else "no_rain"
    )

    # 9. transform 'vehicle_driver_relation'
    df["vehicle_driver_relation"] = df["vehicle_driver_relation"].str.lower()
    df["vehicle_driver_relation"] = df["vehicle_driver_relation"].apply(
        lambda x: "unknown" if x == "other" else x
    )

    # 10. transform 'accident_severity' # target
    severity_weights = { # Change to  1, 2, 3 for slight, serious, fatal
        'Slight Injury': 1,
        'Serious Injury': 1,
        'Fatal Injury': 1
    }
    df['severity_score'] = df['accident_severity'].map(severity_weights)

    # 11. transform 'peak_hour'
    def get_peak(hour, day_of_week):
        if day_of_week in ["saturday", "sunday"]:
            return 'peak' if 7 <= int(hour) < 15 else "off_peak"
        else: 
            return "peak" if 7 <= int(hour) < 19 else "off_peak"
    for row in df.itertuples():
        df.at[row.Index, "peak_hour"] = get_peak(row.hour, row.day_of_week)

    # Group and create 'incident_count'
    group_cols = [
        # 'peak_hour', # Decided to remove this for now
        'types_of_junction',
        'area_accident_occured', 
        'lanes_or_medians',
        'weather_conditions',
        'day_of_week',
        'age_band_of_driver',
        'type_of_vehicle', 
        # 'vehicle_driver_relation' # Decided to remove this for now
        ]

    # Drop rows with missing values in the specified columns
    df = df.dropna(subset=group_cols)

    # grouped = df.groupby(group_cols).size().reset_index(name='incident_count')
    grouped = df.groupby(group_cols)['severity_score'].sum().reset_index(name='incident_count')
    print("incident counter:", grouped)

    # Export the incident counter dataframe to CSV
    incident_counter_csv_filename = '/opt/airflow/dags/data/incident_counts.csv'
    # grouped.to_csv(incident_counter_csv_filename, index=False)
    print(f"Incident counter data saved to '{incident_counter_csv_filename}'")

    # Merge back features (aggregated view)
    df_grouped = grouped.copy()
    # Merge back the original features to the grouped DataFrame
    # One-hot encode all categorical features
    encoded_features = pd.get_dummies(df_grouped[group_cols])
    X = encoded_features
    y = df_grouped['incident_count']

    # Train regression model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from xgboost import XGBRegressor

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Models to evaluate
    RandomForestRegressor_model = RandomForestRegressor(random_state=42)
    LinearRegression_model = LinearRegression()
    Ridge_model = Ridge(alpha=1.0)
    DecisionTreeRegressor_model = DecisionTreeRegressor(random_state=42)
    GradientBoostingRegressor_model = GradientBoostingRegressor(random_state=42)
    XGBRegressor_model = XGBRegressor(random_state=42)

    # Fit models
    RandomForestRegressor_model.fit(X_train, y_train)
    LinearRegression_model.fit(X_train, y_train)
    Ridge_model.fit(X_train, y_train)
    DecisionTreeRegressor_model.fit(X_train, y_train)
    GradientBoostingRegressor_model.fit(X_train, y_train)
    XGBRegressor_model.fit(X_train, y_train)

    models = {
        "RandomForestRegressor": RandomForestRegressor_model,
        "LinearRegression": LinearRegression_model,
        "Ridge": Ridge_model,
        "DecisionTreeRegressor": DecisionTreeRegressor_model,
        "GradientBoostingRegressor": GradientBoostingRegressor_model,
        "XGBRegressor": XGBRegressor_model
    }

    # Store results
    model_performance = []

    # Loop through each model and evaluate
    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        model_performance.append({
            "Model": name,
            "RMSE": rmse,
            "R2": r2
        })

    performance_df = pd.DataFrame(model_performance)

    # Find the best model
    # Sort by lowest RMSE and then highest R2
    performance_df = performance_df.sort_values(by=["RMSE", "R2"], ascending=[True, False])
    best_model_name = performance_df.iloc[0]["Model"]
    best_model = models[best_model_name]
    
    print(f"Best model: {best_model} with RMSE: {performance_df.iloc[0]['RMSE']} and R^2: {performance_df.iloc[0]['R2']}")
    csv_filename = '/opt/airflow/dags/data/models/model_performance.csv'
    
    performance_df.to_csv(csv_filename, index=False)
    # Feature Importance 
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False)

    # Save top features
    top_features_df = pd.DataFrame({'Feature': top_features.index, 'Importance': top_features.values})
    csv_filename = '/opt/airflow/dags/data/top_incident_rate_features.csv'
    # top_features_df.to_csv(csv_filename, index=False)

    # Save min and max values of target for min-max normalization
    min_rate = y.min()
    max_rate = y.max()
    p95 = y.quantile(0.95)
    range_path = "/opt/airflow/dags/data/incident_rate_range.json"

    with open(range_path, "w") as f:
        json.dump({"min": float(min_rate), "max": float(max_rate), "p95": float(p95)}, f)
        

    # --- Save the trained model ---
    model_path = '/opt/airflow/dags/data/models/incident_rate_model.pkl'
    joblib.dump(best_model, model_path)

    # --- Save the one-hot encoded column names (feature structure) ---
    feature_path = '/opt/airflow/dags/data/models/incident_rate_features.pkl'
    joblib.dump(list(X.columns), feature_path)

with DAG("train_model",
         default_args=default_args,
         schedule_interval="@daily",
         start_date=datetime(2025, 1, 1),
         catchup=False,
         tags=["traffic", "ml","IS3107"]) as dag:

    train_model = PythonOperator(
        task_id="train_model",
        # python_callable=train_model1,
        python_callable=train_model2,
    )


    train_model 
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import create_engine
import os

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_junction_data():
    """Load the junction data from the CSV file"""
    junction_file = "/opt/airflow/dags/data/junctions_from_geojson.csv"
    if not os.path.exists(junction_file):
        raise FileNotFoundError(f"Junction data file not found: {junction_file}")
    
    # Load the junction data
    junctions_df = pd.read_csv(junction_file)
    
    # Convert to GeoDataFrame with Point geometry
    geometry = [Point(xy) for xy in zip(junctions_df['junction_longitude'], junctions_df['junction_latitude'])]
    junctions_gdf = gpd.GeoDataFrame(junctions_df, geometry=geometry, crs="EPSG:4326")
    
    # Save the GeoDataFrame for later use
    junctions_gdf.to_pickle("/opt/airflow/dags/data/junctions_geodataframe.pkl")
    
    print(f"Loaded {len(junctions_gdf)} junctions")
    return len(junctions_gdf)

def load_traffic_data():
    """Load traffic data from the database"""
    # Replace with your actual database connection details
    db_connection = os.environ.get('DB_CONNECTION', 'postgresql://airflow:airflow@postgres/airflow')
    engine = create_engine(db_connection)
    
    # Adjust the query based on your actual table structure
    query = """
    SELECT 
        timestamp, 
        latitude, 
        longitude, 
        traffic_level,
        incident_type,
        -- Add other relevant columns
    FROM traffic_data
    WHERE timestamp >= NOW() - INTERVAL '7 days'
    """
    
    traffic_df = pd.read_sql(query, engine)
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(traffic_df['longitude'], traffic_df['latitude'])]
    traffic_gdf = gpd.GeoDataFrame(traffic_df, geometry=geometry, crs="EPSG:4326")
    
    # Save the GeoDataFrame for later use
    traffic_gdf.to_pickle("/opt/airflow/dags/data/traffic_geodataframe.pkl")
    
    print(f"Loaded {len(traffic_gdf)} traffic records")
    return len(traffic_gdf)

def load_rainfall_data():
    """Load rainfall data from the database"""
    # Replace with your actual database connection details
    db_connection = os.environ.get('DB_CONNECTION', 'postgresql://airflow:airflow@postgres/airflow')
    engine = create_engine(db_connection)
    
    # Adjust the query based on your actual table structure
    query = """
    SELECT 
        timestamp, 
        latitude, 
        longitude, 
        rainfall_amount,
        -- Add other relevant columns
    FROM rainfall_data
    WHERE timestamp >= NOW() - INTERVAL '7 days'
    """
    
    rainfall_df = pd.read_sql(query, engine)
    
    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(rainfall_df['longitude'], rainfall_df['latitude'])]
    rainfall_gdf = gpd.GeoDataFrame(rainfall_df, geometry=geometry, crs="EPSG:4326")
    
    # Save the GeoDataFrame for later use
    rainfall_gdf.to_pickle("/opt/airflow/dags/data/rainfall_geodataframe.pkl")
    
    print(f"Loaded {len(rainfall_gdf)} rainfall records")
    return len(rainfall_gdf)

def merge_data_with_junctions():
    """Merge traffic and rainfall data with junction data"""
    # Load the saved GeoDataFrames
    junctions_gdf = pd.read_pickle("/opt/airflow/dags/data/junctions_geodataframe.pkl")
    traffic_gdf = pd.read_pickle("/opt/airflow/dags/data/traffic_geodataframe.pkl")
    rainfall_gdf = pd.read_pickle("/opt/airflow/dags/data/rainfall_geodataframe.pkl")
    
    # Define the radius for spatial join (in meters)
    # 20 meters is approximately 0.0002 degrees in latitude/longitude
    radius = 0.0002
    
    # Function to find the nearest junction for each point
    def find_nearest_junction(point, junctions_gdf, radius):
        # Create a buffer around the point
        buffer = point.buffer(radius)
        
        # Find junctions that intersect with the buffer
        nearby_junctions = junctions_gdf[junctions_gdf.geometry.intersects(buffer)]
        
        if len(nearby_junctions) == 0:
            return None
        
        # Calculate distances to all nearby junctions
        distances = nearby_junctions.geometry.apply(lambda x: point.distance(x))
        
        # Find the closest junction
        closest_idx = distances.idxmin()
        closest_junction = nearby_junctions.loc[closest_idx]
        
        return closest_junction
    
    # Process traffic data
    traffic_features = []
    for idx, row in traffic_gdf.iterrows():
        point = row.geometry
        nearest_junction = find_nearest_junction(point, junctions_gdf, radius)
        
        if nearest_junction is not None:
            # Extract features from the junction
            junction_features = {
                'timestamp': row['timestamp'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'traffic_level': row['traffic_level'],
                'incident_type': row.get('incident_type', 'unknown'),
                'junction_type': nearest_junction['junction_type'],
                'num_roads': nearest_junction['num_roads'],
                'has_unnamed_road': 'Unnamed Road' in nearest_junction['road_name'],
                'has_expressway': 'expressway' in nearest_junction['road_type'].lower() if isinstance(nearest_junction['road_type'], str) else False,
                'avg_lanes': nearest_junction['lanes'] if isinstance(nearest_junction['lanes'], (int, float)) else None,
                'max_speed': nearest_junction['maxspeed'] if isinstance(nearest_junction['maxspeed'], (int, float)) else None,
                'road_surface': nearest_junction['surface'],
                'road_ref': nearest_junction['ref']
            }
            traffic_features.append(junction_features)
    
    # Process rainfall data
    rainfall_features = []
    for idx, row in rainfall_gdf.iterrows():
        point = row.geometry
        nearest_junction = find_nearest_junction(point, junctions_gdf, radius)
        
        if nearest_junction is not None:
            # Extract features from the junction
            junction_features = {
                'timestamp': row['timestamp'],
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'rainfall_amount': row['rainfall_amount'],
                'junction_type': nearest_junction['junction_type'],
                'num_roads': nearest_junction['num_roads'],
                'has_unnamed_road': 'Unnamed Road' in nearest_junction['road_name'],
                'has_expressway': 'expressway' in nearest_junction['road_type'].lower() if isinstance(nearest_junction['road_type'], str) else False,
                'avg_lanes': nearest_junction['lanes'] if isinstance(nearest_junction['lanes'], (int, float)) else None,
                'max_speed': nearest_junction['maxspeed'] if isinstance(nearest_junction['maxspeed'], (int, float)) else None,
                'road_surface': nearest_junction['surface'],
                'road_ref': nearest_junction['ref']
            }
            rainfall_features.append(junction_features)
    
    # Convert to DataFrames
    traffic_features_df = pd.DataFrame(traffic_features)
    rainfall_features_df = pd.DataFrame(rainfall_features)
    
    # Save the feature DataFrames
    traffic_features_df.to_csv("/opt/airflow/dags/data/traffic_with_junctions.csv", index=False)
    rainfall_features_df.to_csv("/opt/airflow/dags/data/rainfall_with_junctions.csv", index=False)
    
    # Combine traffic and rainfall data
    combined_features = pd.concat([traffic_features_df, rainfall_features_df], ignore_index=True)
    combined_features.to_csv("/opt/airflow/dags/data/combined_features.csv", index=False)
    
    print(f"Created feature dataset with {len(combined_features)} records")
    return len(combined_features)

# Create the DAG
with DAG(
    "feature_engineering_dag",
    default_args=default_args,
    description="Feature engineering pipeline that integrates junction data with traffic and rainfall data",
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=["feature_engineering", "traffic", "rainfall", "junctions"]
) as dag:
    
    load_junction_task = PythonOperator(
        task_id="load_junction_data",
        python_callable=load_junction_data,
    )
    
    load_traffic_task = PythonOperator(
        task_id="load_traffic_data",
        python_callable=load_traffic_data,
    )
    
    load_rainfall_task = PythonOperator(
        task_id="load_rainfall_data",
        python_callable=load_rainfall_data,
    )
    
    merge_data_task = PythonOperator(
        task_id="merge_data_with_junctions",
        python_callable=merge_data_with_junctions,
    )
    
    # Define task dependencies
    [load_junction_task, load_traffic_task, load_rainfall_task] >> merge_data_task 
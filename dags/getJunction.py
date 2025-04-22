import json
from logging import Logger
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import requests
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
import pandas as pd


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 3, 24),
}
def download_all_singapore_roads():
    overpass_url = "http://overpass-api.de/api/interpreter"

    query = """
    [out:json][timeout:180];
    (
      way["highway"](1.137, 103.601, 1.460, 104.120);
    );
    out geom;
    """

    response = requests.get(overpass_url, params={'data': query})

    if response.status_code != 200:
        raise Exception(f"Overpass API request failed: {response.status_code}, {response.text}")

    data = response.json()
    elements = data.get("elements", [])
    features = []

    for element in elements:
        if element["type"] == "way" and "geometry" in element:
            coordinates = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
            line = LineString(coordinates)
            
            # Extract road properties
            tags = element.get("tags", {})
            road_properties = {
                "name": tags.get("name", "Unnamed Road"),
                "highway_type": tags.get("highway", "unknown"),
                "oneway": tags.get("oneway", "no"),
                "lanes": tags.get("lanes", "unknown"),
                "maxspeed": tags.get("maxspeed", "unknown"),
                # "ref": tags.get("ref", ""),  # Road reference number
                "surface": tags.get("surface", "unknown")
            }
            
            features.append({
                "type": "Feature",
                "geometry": line.__geo_interface__,
                "properties": road_properties
            })

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    with open("/opt/airflow/dags/data/singapore_highways.geojson", "w") as geojson_file:
        geojson_file.write(json.dumps(geojson_data, indent=2))

    print(f"Saved {len(features)} road features as GeoJSON.")
    return len(features)
# Step 4: Junction classification logic
def angle_between(p1, p2):
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    angle = np.degrees(np.arctan2(dy, dx)) % 360
    return angle

def classify_junction(angles):
    n = len(angles)
    if n < 2:
        return "No Junction"
    angles = sorted(angles)
    if n == 2:
        return "No Junction"
    elif n == 3:
        diffs = [abs((angles[i] - angles[j]) % 360) for i in range(n) for j in range(i+1, n)]
        if any(abs(d - 180) < 30 for d in diffs):
            return "T Shape"
        elif all(abs(d - 120) < 30 for d in diffs):
            return "Y Shape"
        else:
            return "Other"
    elif n == 4:
        diffs = [(angles[i+1] - angles[i]) % 360 for i in range(3)]
        diffs.append((360 + angles[0] - angles[-1]) % 360)
        if all(abs(d - 90) < 30 for d in diffs):
            return "X Shape"
        else:
            return "Crossing"
    elif n > 4:
        return "O Shape"
    return "Unknown"

def extract_junction_type():
    # Step 1: Load GeoJSON
    gdf = gpd.read_file("/opt/airflow/dags/data/singapore_highways.geojson")

    # Step 2: Extract all node coordinates from line endpoints
    point_map = {}  # key = Point(x, y), value = list of road indices

    for idx, geom in gdf.geometry.items():
        if not isinstance(geom, LineString):
            continue
        start = geom.coords[0]
        end = geom.coords[-1]
        for pt in [start, end]:
            pt_key = tuple(np.round(pt, 6))  # rounded for tolerance
            point_map.setdefault(pt_key, []).append(idx)

    # Step 3: Find candidate junctions (nodes connected to ≥3 roads)
    junction_coords = [pt for pt, roads in point_map.items() if len(roads) >= 3]

    # Step 5: Build junction DataFrame
    junction_data = []

    for coord in junction_coords:
        connected_roads = point_map[coord]
        angles = []
        road_info = []

        for road_idx in connected_roads:
            line = gdf.geometry[road_idx]
            coords = line.coords
            start_coord = coords[0]
            end_coord = coords[-1]
            
            if tuple(np.round(start_coord, 6)) == coord:
                other = coords[1]
            else:
                other = coords[-2]
            angles.append(angle_between(coord, other))
            
            # Get road properties directly from the GeoDataFrame
            # The properties are stored as columns in the GeoDataFrame, not as a nested "properties" column
            road_info.append({
                "name": gdf.iloc[road_idx].get("name", "Unnamed Road"),
                "type": gdf.iloc[road_idx].get("highway_type", "unknown"),
                "oneway": gdf.iloc[road_idx].get("oneway", "no"),
                "lanes": gdf.iloc[road_idx].get("lanes", "unknown"),
                "maxspeed": gdf.iloc[road_idx].get("maxspeed", "unknown"),
                "ref": gdf.iloc[road_idx].get("ref", ""),
                "surface": gdf.iloc[road_idx].get("surface", "unknown"),
                "start_lon": start_coord[0],
                "start_lat": start_coord[1],
                "end_lon": end_coord[0],
                "end_lat": end_coord[1]
            })

        jtype = classify_junction(angles)
        
        # Flatten the data structure
        for road in road_info:
            junction_data.append({
                "junction_longitude": coord[0],
                "junction_latitude": coord[1],
                "junction_type": jtype,
                "num_roads": len(connected_roads),
                "road_name": road["name"],
                "road_type": road["type"],
                "oneway": road["oneway"],
                "lanes": road["lanes"],
                "maxspeed": road["maxspeed"],
                "ref": road["ref"],
                "surface": road["surface"],
                "road_start_lon": road["start_lon"],
                "road_start_lat": road["start_lat"],
                "road_end_lon": road["end_lon"],
                "road_end_lat": road["end_lat"]
            })

    junctions_df = pd.DataFrame(junction_data)
    junctions_df.to_csv("/opt/airflow/dags/data/junctions_from_geojson.csv", index=False)
    print("✅ Exported junctions_from_geojson.csv with road information")

with DAG("generate_junctions_from_geojson",
         default_args=default_args,
         schedule_interval=None,
         catchup=False,
         tags=["traffic", "geocoding","IS3107"]) as dag:

    download_all_singapore_roads = PythonOperator(
        task_id="download_all_singapore_roads",
        python_callable=download_all_singapore_roads,
        op_kwargs={'lat': 1.3521, 'lon': 103.8198}  # Example: Singapore central coordinates
    )
    extract_junction_type = PythonOperator(
        task_id="extract_junction_type",
        python_callable=extract_junction_type,
    )

    download_all_singapore_roads >> extract_junction_type

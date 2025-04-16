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
            features.append({
                "type": "Feature",
                "geometry": line.__geo_interface__,
                "properties": element.get("tags", {})
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
        return "no_junction"
    angles = sorted(angles)
    if n == 2:
        return "no_junction"
    elif n == 3:
        diffs = [abs((angles[i] - angles[j]) % 360) for i in range(n) for j in range(i+1, n)]
        if any(abs(d - 180) < 30 for d in diffs):
            return "t_shape"
        elif all(abs(d - 120) < 30 for d in diffs):
            return "t_shape"
        else:
            return "other"
    elif n == 4:
        diffs = [(angles[i+1] - angles[i]) % 360 for i in range(3)]
        diffs.append((360 + angles[0] - angles[-1]) % 360)
        if all(abs(d - 90) < 30 for d in diffs):
            return "x_shape"
        else:
            return "crossing"
    elif n > 4:
        return "o_shape"
    return "unknown"

def extract_road_metadata():
    gdf = gpd.read_file("/opt/airflow/dags/data/singapore_highways.geojson")

    road_records = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if not isinstance(geom, LineString):
            continue

        name = row.get("name", "Unnamed Road")
        oneway_val = str(row.get("oneway", "no")).lower()
        is_two_way = not (oneway_val in ("yes", "true", "1"))

        coords = list(geom.coords)
        start = coords[0]
        end = coords[-1]

        road_records.append({
            "road_name": name,
            "start_lon": start[0],
            "start_lat": start[1],
            "end_lon": end[0],
            "end_lat": end[1],
            "two_way": is_two_way
        })

    df = pd.DataFrame(road_records)
    df.to_csv("/opt/airflow/dags/data/road_metadata.csv", index=False)
    print("✅ Exported road_metadata.csv")



def extract_junction_type() :
   # Step 1: Load GeoJSON
  gdf = gpd.read_file("/opt/airflow/dags/data/singapore_highways.geojson")  # replace with your filename

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

      for road_idx in connected_roads:
          line = gdf.geometry[road_idx]
          coords = line.coords
          if tuple(np.round(coords[0], 6)) == coord:
              other = coords[1]
          else:
              other = coords[-2]
          angles.append(angle_between(coord, other))

      jtype = classify_junction(angles)
      junction_data.append({
          "longitude": coord[0],
          "latitude": coord[1],
          "junction_type": jtype,
          "num_roads": len(connected_roads),
      })
        #   "connected_road_indices": connected_roads

  junctions_df = pd.DataFrame(junction_data)
  junctions_df.to_csv("/opt/airflow/dags/data/junctions_from_geojson.csv", index=False)
  print("✅ Exported junctions_from_geojson.csv")

with DAG("get_road_type_dag",
         default_args=default_args,
         schedule_interval="@daily",
         start_date=datetime(2025, 1, 1),
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

    extract_road_metadata = PythonOperator(
        task_id="extract_road_metadata",
        python_callable=extract_road_metadata,
    )


    download_all_singapore_roads >> extract_junction_type >> extract_road_metadata

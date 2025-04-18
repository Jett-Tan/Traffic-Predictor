import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
# Paths to model and encoders
MODEL_PATH = "./dags/data/accident_severity_model.pkl"
ENCODERS_PATH = "./dags/tmp/encoders.pkl"  # Or wherever you saved them

# URLS
GETTOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"

GETTOKEN_PAYLOAD = {
  "email": "e1122898@u.nus.edu",
  "password": "P@ssw0rd1234"
}

global token
# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load encoders
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

app = Flask(__name__)

def get_token():
    response = requests.request("POST", GETTOKEN_URL, json=GETTOKEN_PAYLOAD)
    if response.status_code == 200:
        token = response.json()["access_token"]
        return response.json()["access_token"]
    else:
        raise Exception("Failed to get token")
    
def get_lat_lon_from_location(location_name):
    token = get_token()
    headers = {"Authorization": "Bearer " + token}
        
    url = f"https://www.onemap.gov.sg/api/common/elastic/search"
    params = {
        "searchVal": location_name,
        "returnGeom":"Y",
        "getAddrDetails":"Y",
        "pageNum":1
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if not data:
        return None, None
    return float(data["results"][0]["LATITUDE"]), float(data["results"][0]["LONGITUDE"])

def extract_route_from_start_and_end(start_lat:float, start_lon:float, end_lat:float, end_lon:float):
    token = get_token()
    headers = {"Authorization": "Bearer " + token}
        
    url = f"https://www.onemap.gov.sg/api/public/routingsvc/route"
    params = {        
        "start": str(start_lat) + "," + str(start_lon),
        "end": str(end_lat) + "," + str(end_lon),
        "routeType":"drive",
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if not data:
        return None, None
    if not data["route_instructions"]:
        return None, None
    instructions = []
    for i in range(len(data["route_instructions"])):
        instructions.append({
            "direction":data["route_instructions"][i][0],
            "road_name":data["route_instructions"][i][1],
            "longitude":data["route_instructions"][i][3].split(",")[0],
            "latitude":data["route_instructions"][i][3].split(",")[1]
        })
    return instructions

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

def map_to_geoJson(route):
    # Extract features from the route data
    # This is a placeholder function. You need to implement the actual feature extraction logic.
    returnValues = []
    # {
    #   "area_accident_occured":'residential'|'service'|'living_street'|'road'|'tertiary', 
    #   "lanes_or_medians": 'two_way'|'other'|'one_way'|'unknown', 
    #   "types_of_junction": 'no_junction'|'y_shape'|'crossing'|'o_shape'|'other'|'unknown'|'t_shape'|'x_shape', 
    #   "weather_conditions": 'no_rain'|'rain'
    # }
    for i in range(len(route)):
        overpass_url = "http://overpass-api.de/api/interpreter"

        query = f"""
          [out:json][timeout:18];
          way(around:10,{route[i]["longitude"]}, {route[i]["latitude"]})["highway"];
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
        returnValues.append({
            "road_name": route[i]["road_name"],
            "direction": route[i]["direction"],
            "latitude": route[i]["latitude"],
            "longitude": route[i]["longitude"],
            "geojson_data": geojson_data
        })
    print("returnValues", returnValues)
    return returnValues

def extract_junction_type(gdf) :
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
          "junction_type": jtype
      })
        #   "connected_road_indices": connected_roads

  junctions_df = pd.DataFrame(junction_data)
  junctions_df.to_csv("/opt/airflow/dags/data/junctions_from_geojson.csv", index=False)
  print("✅ Exported junctions_from_geojson.csv")


@app.route("/predict", methods=["POST"])
def predict():
    print("Received request for prediction")
    print(request.json)
    if not request.json:
        return jsonify({"error": "Invalid input"}), 400
    data = request.json
    start = data.get("start")
    end = data.get("end")

    if not start or not end:
        return jsonify({"error": "Missing start or end"}), 400
    
    # Convert to lat/lon
    start_lat, start_lon = get_lat_lon_from_location(start)
    end_lat, end_lon = get_lat_lon_from_location(end)
    if None in (start_lat, start_lon, end_lat, end_lon):
        return jsonify({"error": "Could not geocode one or both locations"}), 400
  
    routes = extract_route_from_start_and_end(start_lat, start_lon, end_lat, end_lon)
    
    if not routes:
        return jsonify({"error": "Could not extract route"}), 400
    
    geoJson = map_to_geoJson(routes)


    return jsonify(geoJson)
    # input_encoded = {}
    # for key, value in data.items():
    #     if key in encoders:
    #         input_encoded[key] = encoders[key].transform([value])[0]
    #     else:
    #         input_encoded[key] = value

    # input_df = pd.DataFrame([input_encoded])
    # prediction = model.predict(input_df)[0]
    # result = encoders["accident_severity"].inverse_transform([prediction])[0]
    # return jsonify({"predicted_accident_severity": result})
    return jsonify({"predicted_accident_severity": "result"})

if __name__ == "__main__":
    app.run(debug=True)

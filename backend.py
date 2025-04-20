import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import requests
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from sklearn.neighbors import BallTree
import joblib
import json
#
JUNCTIONS_CSV = "./dags/data/junctions_from_geojson.csv"
# RAINFALL_CSV = "./dags/data/rainfall/rainfall_data_postgres.csv"
IMPORTANCE_CSV = "./dags/data/top_incident_rate_features.csv"
# IMPORTANCE_CSV = "./dags/data/top_feature_importances.csv"
MODEL_PATH = './dags/data/models/incident_rate_model.pkl'
FEATURE_PATH = './dags/data/models/incident_rate_features.pkl'
# URLS
GETTOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"

GETTOKEN_PAYLOAD = {
  "email": "e1122898@u.nus.edu",
  "password": "P@ssw0rd1234"
}

global token
# Load csv
try:
    junctions_df = pd.read_csv(JUNCTIONS_CSV)
    # rainfall_df = pd.read_csv(RAINFALL_CSV)
    importance_df = pd.read_csv(IMPORTANCE_CSV, index_col='Feature')
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH)
    with open("./dags/data/incident_rate_range.json") as f:
        rate_range = json.load(f)
except FileNotFoundError as e:
    print(f"Error: One or both CSV files not found: {e}")
    
print("Running backend localhost:5000") 
# Prepare BallTree only once (use radians)
# Build the 2D array of lat/lon in radians
coords_rad_junction = np.radians(junctions_df[["junction_latitude", "junction_longitude"]].values)
# coords_rad_rainfall = np.radians(rainfall_df[["latitude", "longitude"]].values)

# Then build the BallTree
junction_tree = BallTree(coords_rad_junction, metric="haversine")
# rainfall_tree = BallTree(coords_rad_rainfall, metric="haversine")

app = Flask(__name__)

def get_token():
    response = requests.request("POST", GETTOKEN_URL, json=GETTOKEN_PAYLOAD)
    if response.status_code == 200:
        token = response.json()["access_token"]
        return response.json()["access_token"]
    else:
        raise Exception("Failed to get token")
    
def get_lat_lon_from_location(location_name):
    print("Running get_lat_lon_from_location")
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
    try:
        return float(data["results"][0]["LATITUDE"]), float(data["results"][0]["LONGITUDE"])
    except (IndexError, KeyError):
        raise Exception("Failed to extract latitude and longitude from response")

def extract_route_from_start_and_end(start_lat:float, start_lon:float, end_lat:float, end_lon:float):
    print("Running extract_route_from_start_and_end")
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
            "latitude":data["route_instructions"][i][3].split(",")[0],
            "longitude":data["route_instructions"][i][3].split(",")[1],
        })
    print(instructions[0])
    return instructions

def get_weather_data():
    
    # === Step 1: Fetch data from API ===
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error:", response.status_code)
        exit()

    data = response.json()

    # === Step 2: Extract timestamp and readings ===
    timestamp = data["items"][0]["timestamp"]  # ISO string like "2025-03-29T20:45:00+08:00"
    readings = data["items"][0]["readings"]
    stations = {s["id"]: s for s in data["metadata"]["stations"]}

    # === Step 3: Combine readings with metadata ===
    combined = []
    for r in readings:
        station = stations.get(r["station_id"], {})
        combined.append({
            "station": station.get("name", r["station_id"]),
            "latitude": station.get("location", {}).get("latitude"),
            "longitude": station.get("location", {}).get("longitude"),
            "rainfall": r["value"],
            "collected_at": timestamp  # Add timestamp to each row
        })

    df = pd.DataFrame(combined)

    return df

def add_features_to_routes(routes,driver_age, vehicle_type, day_of_week):
    print("Running get_all_scores")
    # Load the CSV file into a DataFrame
    scores_included = []
    for step in routes:
        newStep = add_features_to_route(step)
        scores_included.append(newStep)

    for step in scores_included:
        step["driver_age"] = driver_age
        step["vehicle_type"] = vehicle_type
        step["day_of_week"] = day_of_week
    scores_included_return = []   

    for step in scores_included:
        step = get_nearest_rainfall_score(step)
        scores_included_return.append(step)
    return scores_included_return

def add_features_to_route(step, max_distance_m=10): 
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
    highway_classification = {
        "motorway": "unknown",  # Not specified in your new classification
        "trunk": "unknown",     # Same as above
        "primary": "unknown",
        "secondary": "living_street",
        "tertiary": "living_street",
        "unclassified": "unknown",
        "residential": "residential",
        "living_street": "living_street",
        "service": "service",
        "motorway_link": "unknown",
        "trunk_link": "unknown",
        "primary_link": "unknown",
        "secondary_link": "unknown",
        "tertiary_link": "unknown",
        "road": "unknown",
        "construction": "unknown",
        "services": "service",  # Closest fit based on use for access roads
        "bus_guideway": "unknown",
        "raceway": "unknown",
        "escape": "unknown",
        "corridor": "unknown"
    }
    lat = round(float(step["latitude"]), 4)
    lon = round(float(step["longitude"]), 4)
    
    point_rad = np.radians([[lat,lon]])
    dist, idx = junction_tree.query(point_rad, k=1)
    dist_m = dist[0][0] * 6371000  # convert radians to meters
    
    if dist_m <= max_distance_m:
        # print(f"Processing coordinates: {lat}, {lon}, {junctions_df.iloc[idx[0][0]]['junction_latitude']}, { junctions_df.iloc[idx[0][0]]['junction_longitude']}")
        step['junction'] = types_of_junction_to_types_of_junction[junctions_df.iloc[idx[0][0]]["junction_type"]]
        step['lanes_or_medians'] = "one_way" if junctions_df.iloc[idx[0][0]]["oneway"].lower() == "yes" else "two_way"
        step['area_accident_occured'] = highway_classification[junctions_df.iloc[idx[0][0]]["road_type"]]
    else:
        step['junction'] = "unknown"
        step['lanes_or_medians'] = "unknown"
        step['area_accident_occured'] = "unknown"

    return step

def get_nearest_rainfall_score(step, max_distance_m=2500):
    rainfall_df = get_weather_data()
    coords_rad_rainfall = np.radians(rainfall_df[["latitude", "longitude"]].values)
    rainfall_tree = BallTree(coords_rad_rainfall, metric="haversine")
    lat = round(float(step["latitude"]), 4)
    lon = round(float(step["longitude"]), 4)
    
    point_rad = np.radians([[lat,lon]])
    dist, idx = rainfall_tree.query(point_rad, k=1)
    dist_m = dist[0][0] * 6371000  # convert radians to meters
    
    if dist_m <= max_distance_m:
        # print(f"Processing coordinates: {lat}, {lon}, {rainfall_df.iloc[idx[0][0]]['latitude']}, { rainfall_df.iloc[idx[0][0]]['longitude']}, { rainfall_df.iloc[idx[0][0]]['station']}")
        if rainfall_df.iloc[idx[0][0]]["rainfall"] > 0 :
            step["rainfall"] = "rain"
        else:
            step["rainfall"] = "no_rain"
    else:
        step["rainfall"] = "no_rain"
    return step

def compute_score(routes):
    for step in routes:
        # Initialize safety_rate to 0
        step["safety_rate"] = 0

        # Add scores based on features
        step["safety_rate"] += importance_df.loc[f"types_of_junction_{step['junction']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"lanes_or_medians_{step['lanes_or_medians']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"area_accident_occured_{step['area_accident_occured']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"weather_conditions_{step['rainfall']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"age_band_of_driver_{step['driver_age']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"type_of_vehicle_{step['vehicle_type']}", "Importance"]
        step["safety_rate"] += importance_df.loc[f"day_of_week_{step['day_of_week']}", "Importance"]
    return routes

def preprocess_route(route_dict, feature_columns):
    cleaned = {
        'types_of_junction': route_dict['junction'],
        'area_accident_occured': route_dict['area_accident_occured'].lower().strip().replace(" ", "_"),
        'lanes_or_medians': 'two_way' if 'two' in route_dict['lanes_or_medians'].lower()
                             else 'one_way' if 'double' in route_dict['lanes_or_medians'].lower()
                             else route_dict['lanes_or_medians'].lower().strip().replace(" ", "_"),\
        'weather_conditions': route_dict['rainfall'].lower().strip().replace(" ", "_"),
        'age_band_of_driver': route_dict['driver_age'].lower().strip().replace(" ", "_"),
        'type_of_vehicle': route_dict['vehicle_type'].lower().strip().replace(" ", "_"),
        'day_of_week': route_dict['day_of_week'].lower().strip().replace(" ", "_"),
    }

    df = pd.DataFrame([cleaned])
    df_encoded = pd.get_dummies(df)

    # Add missing columns
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]  # ensure correct order

    return df_encoded

def compute_weighted_risk(route_segments):
    total_weighted_score = 0
    total_weight = 0
    for segment in route_segments:
        score = segment["predicted_safety_rate"]
        weight = segment["normalized_safety_score"]
        total_weighted_score += score * weight
        total_weight += weight
    return total_weighted_score / total_weight if total_weight > 0 else 0

@app.route("/predict", methods=["POST"])
def predict():

    # Check if the request contains JSON data
    try:
        if not request.json:
            return jsonify({"error": "Invalid input"}), 400
        data = request.json
        start = data.get("start")
        end = data.get("end")
        vehicle_type = data.get("vehicle_type")
        driver_age = data.get("driver_age")
        day_of_week = datetime.now().strftime("%A").lower()
    
        if not start:
            return jsonify({"error": "Missing start"}), 400
        if not end:
            return jsonify({"error": "Missing end"}), 400
        if not driver_age:
            return jsonify({"error": "Missing driver_age"}), 400
        if not vehicle_type:
            return jsonify({"error": "Missing vehicle_type "}), 400
        if vehicle_type not in ['lorry', 'car', 'bus', 'other', 'motorcycle']:
            return jsonify({"error": "Invalid vehicle_type"}), 400
        if not isinstance(driver_age, int) or driver_age < 18 or driver_age > 100:
            return jsonify({"error": "Invalid driver_age"}), 400
        if driver_age < 30:
            driver_age = "18-30"
        elif driver_age < 50:
            driver_age = '31-50'
        else:
            driver_age = '>51'
        
        # Convert start and end to lat/lon
        start_lat, start_lon = get_lat_lon_from_location(start)
        end_lat, end_lon = get_lat_lon_from_location(end)
        if None in (start_lat, start_lon, end_lat, end_lon):
            return jsonify({"error": "Could not geocode one or both locations"}), 400
        
        # Extract route from start and end
        routes = extract_route_from_start_and_end(start_lat, start_lon, end_lat, end_lon)
        if not routes:
            return jsonify({"error": "Could not extract route"}), 400
        # add features to the routes
        routes = add_features_to_routes(routes, driver_age, vehicle_type, day_of_week)
        
        # Preprocess the route for prediction
        for i in range(len(routes)):
            tempRoute = preprocess_route(routes[i], feature_columns)  
            raw_score = float(model.predict(tempRoute)[0])
            min_rate = rate_range['min']
            max_rate = rate_range['max']
            p95 = rate_range["p95"]

            # Avoid divide-by-zero
            if p95 > min_rate:
                norm_score = (raw_score - min_rate) / (p95 - min_rate)
                norm_score = min(norm_score, 1.0)  # cap at 1.0
            else:
                norm_score = 0.0
            print(f"Raw score: {raw_score}, Normalized score: {norm_score}")    
            routes[i]['predicted_safety_rate'] = round(raw_score,6)
            routes[i]['normalized_safety_score'] =round(norm_score,6)
            
        
            if norm_score < 0.33:
                routes[i]['risk_label'] = "Low"
            elif norm_score < 0.66:
                routes[i]['risk_label'] = "Medium"
            else:
                routes[i]['risk_label'] = "High"
        
        # Averaging the safety scores for the routes
        total_score = 0
        for score in routes:
            total_score += score["predicted_safety_rate"]
        weighted_score = compute_weighted_risk(routes)
        avg_score = round(total_score / len(routes), 6)
        return jsonify({"routes": routes,"scores": {"average_score" : avg_score,"total_score":total_score, "weighted_score": weighted_score}}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

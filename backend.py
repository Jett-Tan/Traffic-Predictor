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
#
JUNCTIONS_CSV = "./dags/data/junctions_from_geojson.csv"  # Or wherever you saved them

# URLS
GETTOKEN_URL = "https://www.onemap.gov.sg/api/auth/post/getToken"

GETTOKEN_PAYLOAD = {
  "email": "e1122898@u.nus.edu",
  "password": "P@ssw0rd1234"
}

global token
# Load csv
junctions_df = pd.read_csv(JUNCTIONS_CSV)
print(junctions_df.columns)
# Prepare BallTree only once (use radians)
# Build the 2D array of lat/lon in radians
coords_rad = np.radians(junctions_df[["junction_latitude", "junction_longitude"]].values)

# Then build the BallTree
junction_tree = BallTree(coords_rad, metric="haversine")

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
    if not data:
        return None, None
    return float(data["results"][0]["LATITUDE"]), float(data["results"][0]["LONGITUDE"])

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
            "longitude":data["route_instructions"][i][3].split(",")[0],
            "latitude":data["route_instructions"][i][3].split(",")[1]
        })
    return instructions

def get_all_scores(routes):
    print("Running get_all_scores")
    # Load the CSV file into a DataFrame
    scores = []
    for step in routes:
        score = get_nearest_score(step)
        if score is not None:
            scores.append(score)
    return scores

def get_nearest_score(step, max_distance_m=10):
    # print("Running get_nearest_score")
    # print(f"Step: {step}")
    lat = round(float(step["latitude"]), 4)
    lon = round(float(step["longitude"]), 4)
    
    point_rad = np.radians([[lon,lat]])
    dist, idx = junction_tree.query(point_rad, k=1)
    dist_m = dist[0][0] * 6371000  # convert radians to meters
    
    if dist_m <= max_distance_m:
        print(f"Processing coordinates: {lat}, {lon}, {junctions_df.iloc[idx[0][0]]['junction_latitude']}, { junctions_df.iloc[idx[0][0]]['junction_longitude']}")
        return junctions_df.iloc[idx[0][0]]["safety_rate"]
    return None

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
    
    scores = get_all_scores(routes)
    if not scores:
        return jsonify({"error": "No matching safety scores found along the route"}), 400

    avg_score = round(sum(scores) / len(scores), 3)
    return jsonify({"predicted_accident_severity": avg_score})

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, jsonify
import requests
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.neighbors import BallTree

app = Flask(__name__)

# === RainfallMethod ===
def get_weather_data():
    print("running")
    url = "https://api.data.gov.sg/v1/environment/rainfall"
    response = requests.get(url)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return None

    data = response.json()
    timestamp = data["items"][0]["timestamp"]
    readings = data["items"][0]["readings"]
    stations = {s["id"]: s for s in data["metadata"]["stations"]}

    combined = []
    for r in readings:
        station = stations.get(r["station_id"], {})
        combined.append({
            "station": station.get("name", r["station_id"]),
            "latitude": station.get("location", {}).get("latitude"),
            "longitude": station.get("location", {}).get("longitude"),
            "rainfall": r["value"],
            "collected_at": timestamp
        })

    df = pd.DataFrame(combined)
    return df

@app.route("/rainfall")
def rainfall():
    df = get_weather_data()
    if df is None:
        return jsonify({"error": "Failed to fetch rainfall data"}), 500
    return df.to_json(orient="records")


# === TrafficIncidentsMethod ===
def get_traffic_incidents():
    print("Fetching traffic incidents...")
    url = "https://datamall2.mytransport.sg/ltaodataservice/TrafficIncidents"
    headers = {
        "AccountKey": "tLB/oJ67QO+OA992i/dU7Q==",
        "accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return None

    data = response.json()
    incidents = data.get("value", [])
    timestamp = datetime.now().isoformat()

    processed = []
    for incident in incidents:
        processed.append({
            "latitude": incident.get("Latitude"),
            "longitude": incident.get("Longitude"),
            "message": incident.get("Message"),
            "type": incident.get("Type"),
            "retrieved_at": timestamp
        })

    df = pd.DataFrame(processed)
    return df

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

def get_nearest_traffic_incident_type(step, max_distance_m=1000):
    traffic_df = get_traffic_incidents()
    
    if traffic_df is None or traffic_df.empty:
        step["traffic_incident"] = "none"
        return step

    coords_rad_incidents = np.radians(traffic_df[["latitude", "longitude"]].values)
    traffic_tree = BallTree(coords_rad_incidents, metric="haversine")
    
    lat = round(float(step["latitude"]), 4)
    lon = round(float(step["longitude"]), 4)
    point_rad = np.radians([[lat, lon]])
    
    dist, idx = traffic_tree.query(point_rad, k=1)
    dist_m = dist[0][0] * 6371000  # Convert radians to meters

    if dist_m <= max_distance_m:
        step["traffic_incident"] = traffic_df.iloc[idx[0][0]]["type"]
    else:
        step["traffic_incident"] = "none"
    
    return step


@app.route("/test-traffic-incident")
def test_traffic_incident():
    # Sample coordinate (you can change this to test different locations)
    sample_step = {
        "latitude": 1.3895551106,
        "longitude": 103.7506040662
    }

    scored_step = get_nearest_traffic_incident_type(sample_step)
    return jsonify(scored_step)


@app.route("/test-rainfall-score")
def test_rainfall_score():
    # Sample test coordinates (you can change this to test different points)
    sample_step = {
        "latitude": 1.3521,
        "longitude": 103.8198
    }

    scored_step = get_nearest_rainfall_score(sample_step)
    return jsonify(scored_step)


@app.route("/traffic-incidents")
def traffic_incidents():
    df = get_traffic_incidents()
    if df is None:
        return jsonify({"error": "Failed to fetch traffic incidents"}), 500
    return df.to_json(orient="records")


if __name__ == "__main__":
    app.run(debug=True)

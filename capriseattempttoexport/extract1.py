import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
import pandas as pd

# Step 1: Load GeoJSON
gdf = gpd.read_file("export1.geojson")  # replace with your filename

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
        "connected_road_indices": connected_roads
    })

junctions_df = pd.DataFrame(junction_data)
junctions_df.to_csv("junctions_from_geojson.csv", index=False)
print("✅ Exported junctions_from_geojson.csv")

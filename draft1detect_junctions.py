import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point

# Step 1: Load the road network graph for Singapore
place = "Singapore"
print("Downloading road network for Singapore...")
G = ox.graph_from_place(place, network_type='drive')
nodes, edges = ox.graph_to_gdfs(G)

# Step 2: Count connections to identify junctions
print("Identifying junction nodes...")
node_counts = edges.groupby("u").size().add(edges.groupby("v").size(), fill_value=0)
junctions = nodes[nodes.index.isin(node_counts[node_counts >= 3].index)].copy()
junctions["degree"] = node_counts[junctions.index]
junctions["junction_type"] = "Unknown"  # Initialize with default

# Step 3: Utility functions
def angle_between(p1, p2):
    """Return angle in degrees between two points (0-360°)."""
    v = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    angle = np.arctan2(v[1], v[0])
    return np.degrees(angle) % 360

def get_connected_edges(node_id, edges_df):
    """Return all edges connected to a node."""
    return edges_df[(edges_df['u'] == node_id) | (edges_df['v'] == node_id)]

def classify_junction(angles):
    """Classify junction based on angle patterns."""
    num_arms = len(angles)
    if num_arms < 2:
        return "No Junction"

    angles = sorted(angles)
    if num_arms == 2:
        return "No Junction"

    elif num_arms == 3:
        diffs = [abs((angles[i] - angles[j]) % 360) for i in range(3) for j in range(i+1, 3)]
        if any(abs(d - 180) < 30 for d in diffs):
            return "T Shape"
        elif all(abs(d - 120) < 30 for d in diffs):
            return "Y Shape"
        else:
            return "Other"

    elif num_arms == 4:
        diffs = [(angles[i+1] - angles[i]) % 360 for i in range(3)]
        diffs.append((360 + angles[0] - angles[-1]) % 360)  # wrap-around
        if all(abs(d - 90) < 30 for d in diffs):
            return "X Shape"
        else:
            return "Crossing"

    elif num_arms > 4:
        return "O Shape"

    return "Unknown"

# Step 4: Classify each junction
print("Classifying junctions...")
for idx, row in junctions.iterrows():
    connected_edges = get_connected_edges(idx, edges)
    angles = []
    lon0, lat0 = row["x"], row["y"]
    for _, edge in connected_edges.iterrows():
        other = edge["v"] if edge["u"] == idx else edge["u"]
        if other in nodes.index:
            lon1, lat1 = nodes.loc[other]["x"], nodes.loc[other]["y"]
            angle = angle_between((lon0, lat0), (lon1, lat1))
            angles.append(angle)
    if angles:
        junctions.at[idx, "junction_type"] = classify_junction(angles)

# Step 5: Rename columns and save output
junctions = junctions.rename(columns={"x": "longitude", "y": "latitude"})
output_file = "sg_road_junctions_classified.csv"
junctions[["longitude", "latitude", "degree", "junction_type"]].to_csv(output_file, index=False)

print(f"Saved classified junctions to {output_file}")

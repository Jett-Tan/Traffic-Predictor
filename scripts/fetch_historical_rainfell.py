import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_rainfall_for_range(start_date_str, end_date_str):
    """
    Fetch daily rainfall data from Data.gov.sg environment/rainfall
    for each day between start_date_str and end_date_str (inclusive).
    Saves separate CSV files to the 'dags/data/rainfall' folder.
    """

    # Convert date strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    # This is your repo's data folder path
    data_folder = "/Users/ruoqiili/Desktop/is3107_dags/Traffic-Predictor/dags/data/rainfall"

    # Loop through each day in the range
    current_date = start_date
    while current_date <= end_date:
        day_str = current_date.strftime("%Y-%m-%d")
        print(f"Fetching data for {day_str}...")

        url = "https://api.data.gov.sg/v1/environment/rainfall"
        params = {"date": day_str}
        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"Error {response.status_code} fetching date={day_str}")
            current_date += timedelta(days=1)
            continue

        data = response.json()

        # The API returns multiple items (one per 5-min interval).
        # We'll combine them all into one DataFrame for that day.
        all_readings = []
        stations_dict = {s["id"]: s for s in data["metadata"]["stations"]}

        for item in data.get("items", []):
            timestamp = item["timestamp"]
            readings = item.get("readings", [])
            for r in readings:
                station = stations_dict.get(r["station_id"], {})
                all_readings.append({
                    "Station": station.get("name", r["station_id"]),
                    "Latitude": station.get("location", {}).get("latitude"),
                    "Longitude": station.get("location", {}).get("longitude"),
                    "Rainfall (mm)": r["value"],
                    "Collected_At": timestamp
                })

        df = pd.DataFrame(all_readings)
        if not df.empty:
            file_date_str = current_date.strftime("%Y%m%d")
            csv_filename = f"{data_folder}/rainfall_data_{file_date_str}.csv"

            df.to_csv(csv_filename, index=False)
            print(f"Saved {len(df)} records to {csv_filename}")
        else:
            print(f"No data returned for {day_str}")

        # Move to the next day
        current_date += timedelta(days=1)

# Usage example:
if __name__ == "__main__":
    # Adjust the dates below as needed.
    start_date_str = "2025-03-29"
    end_date_str   = "2025-04-04"

    fetch_rainfall_for_range(start_date_str, end_date_str)
    print("Done fetching historical rainfall data!")

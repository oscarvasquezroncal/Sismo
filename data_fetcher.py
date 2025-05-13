import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_earthquake_data(min_magnitude=4.5, days=90, save_path="data/raw/earthquakes.csv"):
    print("🔄 Descargando datos sísmicos del USGS...")
    
    endtime = datetime.utcnow()
    starttime = endtime - timedelta(days=days)
    
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": starttime.strftime("%Y-%m-%d"),
        "endtime": endtime.strftime("%Y-%m-%d"),
        "minmagnitude": min_magnitude,
        "orderby": "time"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    records = []
    for feature in data["features"]:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        records.append({
            "time": datetime.utcfromtimestamp(props["time"] / 1000),
            "latitude": coords[1],
            "longitude": coords[0],
            "depth": coords[2],
            "magnitude": props["mag"],
            "place": props["place"]
        })
    
    df = pd.DataFrame(records)
    df.to_csv(save_path, index=False)
    print(f"✅ Datos guardados en: {save_path}")
    return df

if __name__ == "__main__":
    df = fetch_earthquake_data()
    print(df.head())

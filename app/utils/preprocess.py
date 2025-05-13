import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_process_data(csv_path: str, sequence_length: int = 10):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values("time")

    df["time_diff_hr"] = df["time"].diff().dt.total_seconds() / 3600
    df["time_diff_hr"].fillna(df["time_diff_hr"].mean(), inplace=True)

    # Estimamos la energia liberada
    df["energy"] = 10 ** (1.5 * df["magnitude"])

    # Seleccion de columnas importantes 
    features = ["magnitude", "depth", "time_diff_hr", "energy"]
    data = df[features].values

    # Normalizamos
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Genero secuencias
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length][0]) 

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler

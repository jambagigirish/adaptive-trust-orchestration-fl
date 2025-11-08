import numpy as np
import time
import requests
from model_utils import load_local_data, build_model, compute_metadata

SERVER_URL = "https://aggregator.example.com/api/update"
CLIENT_ID = "<client-unique-id>"

def train_local_model(global_weights, local_data, epochs=1, batch_size=32):
    model = build_model()
    model.set_weights(global_weights)
    model.fit(local_data.x, local_data.y, epochs=epochs, batch_size=batch_size)
    delta = model.get_weights() - global_weights
    return delta, model.get_weights()

def compute_local_score(metadata, anomaly_indicator):
    alpha = 0.6
    beta = 0.4
    score_local = alpha * anomaly_indicator + beta * metadata['update_reliability']
    return score_local

def main_loop():
    local_data = load_local_data()
    # initial fetch of global model
    resp = requests.get(f"{SERVER_URL}/get_model?client={CLIENT_ID}")
    global_weights = np.array(resp.json()['weights'])
    trust_score = 1.0
    while True:
        delta, new_weights = train_local_model(global_weights, local_data, epochs=1)
        metadata = compute_metadata(delta, new_weights, local_data)
        anomaly_indicator = metadata['anomaly_score']
        score_local = compute_local_score(metadata, anomaly_indicator)
        payload = {
            "client_id": CLIENT_ID,
            "delta": delta.tolist(),
            "metadata": metadata,
            "local_score": score_local,
            "timestamp": time.time()
        }
        headers = {"Authorization": f"Bearer {CLIENT_ID}-token"}
        requests.post(SERVER_URL + "/upload_update", json=payload, headers=headers)
        # fetch updated global model
        resp = requests.get(f"{SERVER_URL}/get_model?client={CLIENT_ID}")
        global_weights = np.array(resp.json()['weights'])
        time.sleep(60)  # next round in 60s


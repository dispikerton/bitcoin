import pickle

import numpy as np
import uvicorn
from fastapi import FastAPI

from data_preprocessing import load_data, prepare_data
from model import BitcoinPricePredictor

app = FastAPI()

file_path = 'data/bitcoin_2017_to_2023.csv'
window_size = 60
model_path = 'model/model.keras'
scaler_path = 'model/scaler.pkl'
x_test_path = 'data/x_test.npy'
y_test_path = 'data/y_test.npy'

predictor = BitcoinPricePredictor(window_size, 5)


@app.get("/")
async def root():
    return {"message": "Bitcoin Price Prediction API"}


@app.get("/train")
async def train():
    try:
        data = load_data(file_path)
        print("Data loaded.")
        x_train, y_train, x_test, y_test, scaler = prepare_data(data, window_size)
        print("Data prepared.")
        predictor.create_model()
        print("Model created.")
        predictor.train_model(x_train, y_train, save_path=model_path)
        print("Model trained.")
        predictor.evaluate_model(x_test, y_test)
        print("Model evaluated.")
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        np.save(x_test_path, x_test)
        np.save(y_test_path, y_test)
        return {"message": "Model training completed."}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

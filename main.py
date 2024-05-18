import pickle

import numpy as np
import uvicorn
from fastapi import FastAPI
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import load_data, prepare_data
from model import BitcoinPricePredictor

app = FastAPI()

file_path = 'data/bitcoin_2017_to_2023.csv'
window_size = 60
model_path = 'model/model.keras'
scaler_path = 'model/scaler.pkl'
x_test_path = 'data/x_test.npy'
y_test_path = 'data/y_test.npy'
plot_path = 'plot/prediction_plot.png'

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


@app.get("/visualize")
async def visualize():
    try:
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

        predictions = predictor.predict(x_test)

        # Масштабирование только признака 'close'
        close_scaler = MinMaxScaler()
        close_scaler.min_, close_scaler.scale_ = scaler.min_[3], scaler.scale_[3]

        y_test_inverse = close_scaler.inverse_transform(y_test.reshape(-1, 1))
        predictions_inverse = close_scaler.inverse_transform(predictions)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test_inverse, label='Actual')
        plt.plot(predictions_inverse, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        plt.savefig(plot_path)
        plt.close()

        return {"message": f"Visualization completed. Plot saved to {plot_path}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

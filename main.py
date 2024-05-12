from fastapi import FastAPI
from data_preprocessing import load_data, prepare_data
from model import create_model, train_model, evaluate_model, load_model
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
import io
import pickle
import uvicorn
import numpy as np

app = FastAPI()

file_path = 'bitcoin_2017_to_2023.csv'
window_size = 60
model_path = 'model.keras'
scaler_path = 'scaler.pkl'


@app.get("/")
async def root():
    return {"message": "Bitcoin Price Prediction API"}


@app.get("/train")
async def train():
    data = load_data(file_path)
    print("Data loaded.")
    x_train, y_train, x_test, y_test, scaler = prepare_data(data, window_size)
    print("Data prepared.")
    model = create_model(window_size, x_train.shape[2])
    print("Model created.")
    train_model(model, x_train, y_train, save_path=model_path)
    print("Model trained.")
    evaluate_model(model, x_test, y_test)
    print("Model evaluated.")
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    return {"message": "Model training completed.", "x_test": x_test, "y_test": y_test}


@app.get("/load_model")
async def load_trained_model():
    model = load_model(model_path)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return {"message": "Model and scaler loaded successfully.", "model": model, "scaler": scaler}


@app.get("/visualize")
async def visualize(model: dict, scaler: dict, x_test: list, y_test: list):
    model = model["model"]
    scaler = scaler["scaler"]
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    predictions = model.predict(x_test)

    plt.figure(figsize=(12, 6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual')
    plt.plot(scaler.inverse_transform(predictions.reshape(-1, 1)), label='Predicted')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

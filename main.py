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
x_test_path = 'x_test.npy'
y_test_path = 'y_test.npy'


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
        model = create_model(window_size, x_train.shape[2])
        print("Model created.")
        train_model(model, x_train, y_train, save_path=model_path)
        print("Model trained.")
        evaluate_model(model, x_test, y_test)
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
        model = load_model(model_path)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        x_test = np.load(x_test_path)
        y_test = np.load(y_test_path)

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
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

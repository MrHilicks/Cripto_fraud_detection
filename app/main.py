# app/main.py

import os
import sys
import pandas as pd
from fastapi import FastAPI
import uvicorn

from scripts.model_inference import load_model, load_preprocessor, data_preprocess, predict

# === 1. FastAPI app ===
app = FastAPI()

# === 2. POST эндпоинт ===
@app.post("/predict")
def predict_fraud(data):
    df = pd.DataFrame([data.dict()])
    prediction = predict(df)
    return {"prediction": int(prediction[0])}

# === 3. Локальная проверка ===
if __name__ == "__main__":
    TEST_CASES_DIR = os.path.join(BASE_DIR, "Data", "Test Cases")
    TEST_FILES = ["Sample_1.parquet", "Sample_2.parquet", "Sample_3.parquet"]

    print("📦 Локальная проверка FastAPI логики (без сервера)\n")

    for file_name in TEST_FILES:
        file_path = os.path.join(TEST_CASES_DIR, file_name)
        print(f"\n--- 🧪 Предсказание для {file_name} ---")
        try:
            df = pd.read_parquet(file_path)
            prediction = predict(df)
            print(f"✅ Prediction: {int(prediction[0])}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")

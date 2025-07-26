# app/main.py

import os
import sys
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from scripts.model_inference import load_model, load_preprocessor, data_preprocess, predict

# === 1. FastAPI app ===
app = FastAPI()

# === 2. Пути к моделям ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # Папка 1_ML_Project
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_model.cbm")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "feature_preprocessor.pkl")

# === 3. Загружаем модель и препроцессор ===
model = load_model(MODEL_PATH)
feature_preprocessor = load_preprocessor(PREPROCESSOR_PATH)

# === 4. InputData модель ===
class InputData(BaseModel):
    borrow_block_number: int
    borrow_timestamp: float
    wallet_address: str
    first_tx_timestamp: float
    last_tx_timestamp: float
    wallet_age: float
    incoming_tx_count: int
    outgoing_tx_count: int
    net_incoming_tx_count: int
    total_gas_paid_eth: float
    avg_gas_paid_per_tx_eth: float
    risky_tx_count: int
    risky_unique_contract_count: int
    risky_first_tx_timestamp: int
    risky_last_tx_timestamp: int
    risky_first_last_tx_timestamp_diff: int
    risky_sum_outgoing_amount_eth: float
    outgoing_tx_sum_eth: float
    incoming_tx_sum_eth: float
    outgoing_tx_avg_eth: float
    incoming_tx_avg_eth: float
    max_eth_ever: float
    min_eth_ever: float
    total_balance_eth: float
    risk_factor: float
    total_collateral_eth: float
    total_collateral_avg_eth: float
    total_available_borrows_eth: float
    total_available_borrows_avg_eth: float
    avg_weighted_risk_factor: float
    risk_factor_above_threshold_daily_count: float
    avg_risk_factor: float
    max_risk_factor: float
    borrow_amount_sum_eth: float
    borrow_amount_avg_eth: float
    borrow_count: int
    repay_amount_sum_eth: float
    repay_amount_avg_eth: float
    repay_count: int
    borrow_repay_diff_eth: float
    deposit_count: int
    deposit_amount_sum_eth: float
    time_since_first_deposit: float
    withdraw_amount_sum_eth: float
    withdraw_deposit_diff_if_positive_eth: float
    liquidation_count: int
    time_since_last_liquidated: float
    liquidation_amount_sum_eth: float
    market_adx: float
    market_adxr: float
    market_apo: float
    market_aroonosc: float
    market_aroonup: float
    market_atr: float
    market_cci: float
    market_cmo: float
    market_correl: float
    market_dx: float
    market_fastk: float
    market_fastd: float
    market_ht_trendmode: int
    market_linearreg_slope: float
    market_macd_macdext: float
    market_macd_macdfix: float
    market_macd: float
    market_macdsignal_macdext: float
    market_macdsignal_macdfix: float
    market_macdsignal: float
    market_max_drawdown_365d: float
    market_natr: float
    market_plus_di: float
    market_plus_dm: float
    market_ppo: float
    market_rocp: float
    market_rocr: float
    unique_borrow_protocol_count: int
    unique_lending_protocol_count: int

# === 5. POST эндпоинт ===
@app.post("/predict")
def predict_fraud(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = predict(df)
    return {"prediction": int(prediction[0])}

# === 6. Локальная проверка ===
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

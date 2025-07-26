# scripts/model_inference.py

import os
import joblib
import pandas as pd
from catboost import CatBoostClassifier

from scripts.data_preprocessing import DataPreprocessor, FeaturePreprocessor

# === Пути к моделям ===
MODEL_PATH = os.path.join("models", "catboost_model.cbm")
PREPROCESSOR_PATH = os.path.join("models", "feature_preprocessor.pkl")


def load_model(model_path: str) -> CatBoostClassifier:
    """Загрузка модели CatBoost."""
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def load_preprocessor(preprocessor_path: str) -> FeaturePreprocessor:
    """Загрузка сохранённого препроцессора признаков."""
    return FeaturePreprocessor.load(preprocessor_path)


def data_preprocess(df: pd.DataFrame, feature_preprocessor: FeaturePreprocessor) -> pd.DataFrame:
    """Применение полного пайплайна предобработки данных."""
    return DataPreprocessor().data_preprocessing(df, feature_preprocessor)


def predict(df: pd.DataFrame) -> list:
    """Предсказание класса для входных данных."""
    feature_preprocessor = load_preprocessor(PREPROCESSOR_PATH)
    df_processed = data_preprocess(df, feature_preprocessor)
    preds = load_model(MODEL_PATH).predict(df_processed)
    return preds.tolist()


if __name__ == "__main__":
    # === Инференс на тестовых parquet-файлах ===
    TEST_CASES_DIR = os.path.join("Data", "Test Cases")
    test_files = ["Sample_1.parquet", "Sample_2.parquet", "Sample_3.parquet"]

    for file_name in test_files:
        file_path = os.path.join(TEST_CASES_DIR, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            continue

        print(f"\n=== 🔍 Running prediction for: {file_name} ===")
        df = pd.read_parquet(file_path)

        try:
            prediction = predict(df)
            print(f"✅ Prediction class: {prediction}")
        except Exception as e:
            print(f"❌ Error during prediction for {file_name}: {e}")

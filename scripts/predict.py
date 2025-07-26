# scripts/predict.py

import os
import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool


def predict(test_path: str, output_path: str, model_path: str):
    # 1. Загрузка тестовых данных
    test = pd.read_csv(test_path)
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # 2. Загрузка обученной модели
    model = CatBoostClassifier()
    model.load_model(model_path)

    # 3. Предсказания
    y_pred = model.predict(Pool(X_test, y_test))
    y_proba = model.predict_proba(Pool(X_test, y_test))[:, 1]

    # 4. Формирование результата
    result = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    })

    os.makedirs(output_path, exist_ok=True)
    result.to_csv(os.path.join(output_path, "result.csv"), index=False)

    print(f"✅ Предсказания сохранены в {os.path.join(output_path, 'result.csv')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📈 Генерация предсказаний с использованием обученной модели"
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default=os.path.join("Data", "processed", "test.csv"),
        help="📂 Путь к тестовому CSV"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("models", "catboost_model.cbm"),
        help="💾 Путь к обученной модели CatBoost"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join("Data", "processed", "predictions"),
        help="📁 Папка для сохранения result.csv"
    )

    args = parser.parse_args()
    predict(args.test_path, args.output_path, args.model_path)

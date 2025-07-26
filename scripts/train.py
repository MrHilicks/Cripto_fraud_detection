# scripts/train.py

import os
import json
import argparse
import pandas as pd
from catboost import CatBoostClassifier, Pool

from scripts.data_preprocessing import FeaturePreprocessor, DataPreprocessor


def save_sample_json(df_sample: pd.DataFrame, filepath: str):
    """Сохраняем sample в JSON в формате списка словарей."""
    records = df_sample.to_dict(orient="records")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def train_model(train_path, test_path, model_path, preprocessor_path):
    # 1. Загрузка данных
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 2. Создание папки Test Cases (если не существует)
    test_cases_dir = os.path.join(os.path.dirname(test_path), "..", "Test Cases")
    os.makedirs(test_cases_dir, exist_ok=True)

    # 3. Сохраняем 3 тест кейса (parquet + JSON) с разными seed
    for i in range(1, 4):
        sample_df = test.drop(columns=["target"]).sample(n=1, random_state=42 + i)
        parquet_path = os.path.join(test_cases_dir, f"Sample_{i}.parquet")
        json_path = os.path.join(test_cases_dir, f"Sample_{i}.json")

        sample_df.to_parquet(parquet_path)
        save_sample_json(sample_df, json_path)

        print(f"📦 Сохранён тест-кейс #{i} ➜ {parquet_path}, {json_path}")

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # 4. Обучение препроцессора
    feature_preprocessor = FeaturePreprocessor()
    feature_preprocessor.fit(X_train)

    # 5. Преобразование данных
    data_preprocessor = DataPreprocessor()
    X_train_processed = data_preprocessor.data_preprocessing(X_train, feature_preprocessor)
    X_test_processed = data_preprocessor.data_preprocessing(X_test, feature_preprocessor)

    # 6. Обучение модели
    model = CatBoostClassifier(
        iterations=1000,
        eval_metric="F1",
        loss_function="Logloss",
        random_seed=42,
        early_stopping_rounds=30,
        verbose=False,
        allow_writing_files=False,
        task_type="GPU",
        learning_rate=0.151,
        depth=14,
        l2_leaf_reg=7.8457,
        bagging_temperature=0.0443,
        border_count=32,
        random_strength=1.4266,
    )

    model.fit(
        Pool(X_train_processed, y_train),
        eval_set=Pool(X_test_processed, y_test),
        use_best_model=True,
    )

    # 7. Сохранение модели и препроцессора
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)

    model.save_model(model_path)
    feature_preprocessor.save(preprocessor_path)

    print("✅ Модель и препроцессор успешно сохранены.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🚀 Train CatBoost model with preprocessing"
    )

    parser.add_argument(
        "--train_path",
        type=str,
        default=os.path.join("Data", "processed", "train.csv"),
        help="📂 Путь к обучающим данным CSV",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=os.path.join("Data", "processed", "test.csv"),
        help="📂 Путь к тестовым данным CSV",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join("models", "catboost_model.cbm"),
        help="💾 Путь для сохранения модели",
    )
    parser.add_argument(
        "--preprocessor_path",
        type=str,
        default=os.path.join("models", "feature_preprocessor.pkl"),
        help="💾 Путь для сохранения препроцессора",
    )

    args = parser.parse_args()

    train_model(
        args.train_path,
        args.test_path,
        args.model_path,
        args.preprocessor_path,
    )

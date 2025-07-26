# scripts/preprocess.py

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_train_test(raw_path: str, processed_path: str, save_preprocessor=True):
    df = pd.read_parquet(raw_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_path = os.path.join(processed_path, "train.csv")
    test_path = os.path.join(processed_path, "test.csv")

    pd.concat([X_train, y_train], axis=1).to_csv(train_path, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(test_path, index=False)

    print("✅ Тренировочный и тестовый датасеты успешно сохранены.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📦 Preprocess raw dataset and save train/test split"
    )

    parser.add_argument(
        "--raw_path",
        type=str,
        default=os.path.join("Data", "raw", "dataset.parquet"),
        help="📂 Path to raw .parquet file",
    )
    parser.add_argument(
        "--processed_path",
        type=str,
        default=os.path.join("Data", "processed"),
        help="📁 Path to save processed train/test CSVs",
    )
    parser.add_argument(
        "--no_save_preprocessor",
        action="store_true",
        help="🚫 Do not save preprocessor to file",
    )

    args = parser.parse_args()

    preprocess_train_test(
        raw_path=args.raw_path,
        processed_path=args.processed_path,
        save_preprocessor=not args.no_save_preprocessor,
    )

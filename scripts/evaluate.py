# scripts/evaluate.py

import os
import json
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def evaluate(y_true, y_pred, y_proba, metrics_path: str, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Вычисление метрик
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

    # 2. Сохранение метрик в JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()

    # 4. ROC-кривая
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()

    # 5. Precision-Recall кривая
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "pr_curve.png"))
    plt.close()

    # 6. Гистограмма вероятностей
    plt.figure()
    plt.hist(y_proba, bins=50, color="skyblue", edgecolor="black")
    plt.title("Prediction Probabilities Histogram")
    plt.xlabel("Predicted probability")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "probabilities_hist.png"))
    plt.close()

    print("✅ Метрики и графики сохранены")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="📊 Оценка предсказаний модели: метрики и графики"
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=os.path.join("Data", "processed", "predictions", "result.csv"),
        help="📂 CSV с колонками y_true, y_pred, y_proba"
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default=os.path.join("models", "metrics.json"),
        help="💾 Путь для сохранения метрик (JSON)"
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=os.path.join("plots"),
        help="📁 Папка для сохранения графиков"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    y_proba = df["y_proba"].values

    evaluate(y_true, y_pred, y_proba, metrics_path=args.metrics_path, plots_dir=args.plots_dir)

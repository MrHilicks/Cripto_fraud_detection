**README.md**

# 🛡️ Fraud Detection Service

Проект по машинному обучению для классификации мошеннических действий.

## 📂 Структура проекта

Cripto_fraud_detection/
├── app/                 # FastAPI-приложение
├── scripts/             # Предсказания, препроцессинг, инференс
├── models/              # Модели и пайплайны
├── Data/                # Данные (в том числе тест-кейсы)
├── Notebooks/           # Тетрадки JupyterNotebook
├── plots/               # Метрики модели
├── requirements.txt     # Зависимости проекта
├── Dockerfile           # Образ Docker (если нужен)
├── Makefile             # Упрощённые команды
└── README.md            # Описание проекта

## 🚀 Запуск сервиса

```bash
uvicorn app.main:app --reload
```

## 🧪 Тестирование API

```bash
python -m scripts.test_api
```

## 🧰 Предсказания через CLI

```bash
python -m scripts.predict \
  --test_path Data/processed/test.csv \
  --output_path Data/processed/predictions \
  --model_path models/catboost_model.cbm
```

## 🐳 Docker (опционально)

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```
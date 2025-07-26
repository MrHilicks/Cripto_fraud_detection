# scripts/test_api.py

import os
import json
import requests

# === 1. Настройки ===
TEST_CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "Test Cases")
URL = "http://127.0.0.1:8000/predict"

# === 2. Функция для отправки тестов ===
def send_test_cases():
    test_files = [f for f in os.listdir(TEST_CASES_DIR) if f.endswith(".json")]

    if not test_files:
        print("⚠️  В папке 'Test Cases' нет JSON-файлов.")
        return

    for filename in test_files:
        file_path = os.path.join(TEST_CASES_DIR, filename)
        print(f"\n📤 Отправка файла: {filename}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)[0]

            response = requests.post(URL, json=data)
            print(f"✅ Ответ сервиса ({response.status_code}): {response.json()}")
        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")

# === 3. Точка входа ===
if __name__ == "__main__":
    send_test_cases()

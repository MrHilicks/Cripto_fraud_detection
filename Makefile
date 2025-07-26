run:
	uvicorn app.main:app --reload

test:
	python -m scripts.test_api

predict:
	python -m scripts.predict

format:
	black scripts/ app/
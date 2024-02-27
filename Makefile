.PHONY: build test

build:
	docker build -t ct2-wrapper .

inject:
	cd artifacts && ./inject_model_into_image.sh

run-docker:
	docker run --rm -p 8000:8000 --name ct2-model ct2-model

run:
	uvicorn --app-dir src main:app --log-config=src/config.yaml

run-dev:
	uvicorn --app-dir src main:app --reload --log-config=src/config.yaml

test:
	python test/test_doctest.py
	pytest -vvs test/test_pytest.py
	pytest -vvs test/test_api.py
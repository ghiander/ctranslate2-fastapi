.PHONY: build test

build:
	docker build -t ct2-wrapper .

inject:
	cd artifacts && ./inject_model_into_image.sh

run-docker:
	docker run --rm --name ct2-model ct2-model

run:
	uvicorn --app-dir src main:app

run-dev:
	uvicorn --app-dir src main:app --reload

test:
	python test/test_doctest.py
	pytest -vvs test/test_pytest.py
	pytest -vvs test/test_api.py
.PHONY: cleanup
cleanup:
	isort -rc .
	black --target-version=py38 .

.PHONY: lint
lint:
	black --check .
	flake8 .
	pylint autodiff

.PHONY: mypy
mypy:
	# MYPYPATH=$(HOME)/repositories/mypy-data/numpy-mypy mypy --strict dual.py
	mypy --strict --ignore-missing-imports --python-version	3.8 .

.PHONY: test
test:
	python -m pytest -v -s --cov=autodiff

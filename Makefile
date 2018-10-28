.PHONY: pylint
pylint:
	pylint *.py tests/*.py

.PHONY: mypy
mypy:
	# MYPYPATH=$(HOME)/repositories/mypy-data/numpy-mypy mypy --strict dual.py
	mypy --strict --ignore-missing-imports --python-version	3.7 *.py tests/*.py

.PHONY: test
test:
	pytest -v -s tests

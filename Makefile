test: mypy test

pylint:
	pylint *.py

mypy:
	# MYPYPATH=$(HOME)/repositories/mypy-data/numpy-mypy mypy --strict dual.py
	mypy --strict --ignore-missing-imports --python-version	3.7 dual.py

test:
	pytest -v -s tests

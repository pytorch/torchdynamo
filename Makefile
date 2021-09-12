.PHONY: default develop test torchbench format lint setup

PY_FILES := $(wildcard *.py) $(wildcard torchdynamo/*.py) $(wildcard tests/*.py)
C_FILES := $(wildcard torchdynamo/*.c)

default: test

develop:
	python setup.py develop

test: develop
	pytest -v

torchbench: develop
	python torchbench.py

overhead: develop
	python torchbench.py --overhead

format:
	black $(PY_FILES)
	clang-format -i $(C_FILES)

lint:
	flake8 $(PY_FILES)
	clang-tidy $(C_FILES)  -- \
		-I$(shell python -c "from distutils.sysconfig import get_python_inc as X; print(X())")

setup:
	pip install flake8 black pytest

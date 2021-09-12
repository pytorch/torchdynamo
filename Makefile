
default: test

PY_FILES := $(wildcard *.py) $(wildcard torchdynamo/*.py) $(wildcard tests/*.py)
C_FILES := $(wildcard torchdynamo/*.c)

build:
	python setup.py develop

test: build
	pytest

torchbench: test
	python torchbench.py

format:
	black $(PY_FILES)
	clang-format -i $(C_FILES)

lint:
	flake8 $(PY_FILES)
	clang-tidy $(C_FILES) -- -I$(shell python -c "from distutils.sysconfig import get_python_inc as X; print(X())")

setup:
	pip install flake8 black
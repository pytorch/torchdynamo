.PHONY: default develop test torchbench format lint setup clean autotune

PY_FILES := $(wildcard *.py) $(wildcard torchdynamo/*.py) $(wildcard torchdynamo/*/*.py) $(wildcard tests/*.py)
C_FILES := $(wildcard torchdynamo/*.c torchdynamo/*.cpp)

default: develop

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
	clang-format-10 -i $(C_FILES)

lint:
	flake8 $(PY_FILES)
	clang-tidy-10 $(C_FILES)  -- \
		-I$(shell python -c "from distutils.sysconfig import get_python_inc as X; print(X())") \
		$(shell python -c 'from torch.utils.cpp_extension import include_paths; print(" ".join(map("-I{}".format, include_paths())))')

setup:
	pip install flake8 black pytest onnxruntime onnx-tf onnxruntime-gpu tensorflow-gpu

clean:
	python setup.py clean
	rm -rf build torchdynamo.egg-info torchdynamo/*.so

autotune: develop
	python torchbench.py --speedup -n1 --minimum-call-count=2
	python autotune.py
	python torchbench.py --speedup --minimum-call-count=2

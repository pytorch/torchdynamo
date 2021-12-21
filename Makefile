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
	pip install flake8 black pytest onnxruntime-gpu tensorflow-gpu onnx-tf

clean:
	python setup.py clean
	rm -rf build torchdynamo.egg-info torchdynamo/*.so

autotune-cpu: develop
	rm -rf subgraphs
	python torchbench.py --speedup -n1
	python autotune.py
	python torchbench.py --speedup -n50

baseline-cpu: develop
	 rm -f baseline_*.csv
	 python torchbench.py --no-skip -n50 --speedup-ts
	 python torchbench.py --no-skip -n50 --speedup-sr
	 python torchbench.py --no-skip -n50 --speedup-onnx
	 paste -d, baseline_ts.csv baseline_sr.csv baseline_onnx.csv > baseline_all.csv

autotune-gpu: develop
	rm -rf subgraphs
	python torchbench.py --speedup -dcuda --nvfuser -n1
	python autotune.py --nvfuser
	python torchbench.py --speedup -dcuda --nvfuser -n100

baseline-gpu: develop
	 rm -f baseline_*.csv
	 python torchbench.py --no-skip -dcuda -n100 --speedup-ts && mv baseline_ts.csv baseline_nnc.csv
	 python torchbench.py --no-skip -dcuda -n100 --speedup-ts --nvfuser && mv baseline_ts.csv baseline_nvfuser.csv
	 python torchbench.py --no-skip -dcuda -n100 --speedup-trt
	 python torchbench.py --no-skip -dcuda -n100 --speedup-onnx
	 paste -d, baseline_nnc.csv baseline_nvfuser.csv baseline_trt.csv baseline_onnx.csv > baseline_all.csv

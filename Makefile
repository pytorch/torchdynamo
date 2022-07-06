.PHONY: default develop test torchbench format lint setup clean autotune

PY_FILES := $(wildcard *.py) $(wildcard torchdynamo/*.py) $(wildcard torchdynamo/*/*.py) \
            $(wildcard tests/*.py) $(wildcard torchinductor/*.py) $(wildcard torchinductor/*/*.py) \
            $(wildcard benchmarks/*.py) $(wildcard benchmarks/*/*.py)
C_FILES := $(wildcard torchdynamo/*.c torchdynamo/*.cpp)
CLANG_TIDY ?= clang-tidy-10
CLANG_FORMAT ?= clang-format-10

default: develop

develop:
	python setup.py develop

test: develop
	pytest tests

torchbench: develop
	python benchmarks/torchbench.py --fast

overhead: develop
	python benchmarks/torchbench.py --overhead

format:
	isort $(PY_FILES)
	black $(PY_FILES)
	! which $(CLANG_FORMAT) >/dev/null 2>&1 || $(CLANG_FORMAT) -i $(C_FILES)

lint:
	black --check --diff $(PY_FILES)
	isort --check --diff $(PY_FILES)
	flake8 $(PY_FILES)
	! which $(CLANG_TIDY) >/dev/null 2>&1 || $(CLANG_TIDY) $(C_FILES) -- \
		-I`python -c 'from distutils.sysconfig import get_python_inc as X; print(X())'` \
		`python -c 'from torch.utils.cpp_extension import include_paths; print(" ".join(map("-I{}".format, include_paths())))'`

lint-deps:
	grep -E '(black|flake8|isort|click|torch)' requirements.txt | xargs pip install

setup_lint: lint-deps

setup:
	pip install -r requirements.txt

setup_nightly:
	pip install ninja
	pip install --pre torch==1.13.0.dev20220701+cpu --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	pip install -v git+https://github.com/pytorch/functorch.git@6dfa19e61e27012be87c49a249633a58f0cdd024
	pip install -r requirements.txt
	python setup.py develop

clean:
	python setup.py clean
	rm -rf build torchdynamo.egg-info torchdynamo/*.so __pycache__ .pytest_cache .benchmarks *.csv dist

clone-deps:
	(cd .. \
		&& (test -e pytorch || git clone --recursive https://github.com/pytorch/pytorch pytorch) \
		&& (test -e functorch || git clone --recursive https://github.com/pytorch/functorch) \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchtext || git clone --recursive https://github.com/pytorch/text torchtext) \
		&& (test -e torchaudio || git clone --recursive https://github.com/pytorch/audio torchaudio) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/jansel/benchmark torchbenchmark) \
	)

pull-deps:
	(cd ../pytorch        && git pull && git submodule update --init --recursive)
	(cd ../functorch      && git pull && git submodule update --init --recursive)
	(cd ../torchvision    && git pull && git submodule update --init --recursive)
	(cd ../torchtext      && git pull && git submodule update --init --recursive)
	(cd ../torchaudio     && git pull && git submodule update --init --recursive)
	(cd ../detectron2     && git pull && git submodule update --init --recursive)
	(cd ../torchbenchmark && git pull && git submodule update --init --recursive)

build-deps: clone-deps
	# conda env remove --name torchdynamo
	# conda create --name torchdynamo -y python=3.8
	# conda activate torchdynamo
	conda install -y astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake \
        cffi typing_extensions future six requests dataclasses protobuf numba cython
	conda install -y -c pytorch magma-cuda116
	conda install -y -c conda-forge librosa

	make setup && pip uninstall -y torch
	(cd ../pytorch     && python setup.py clean && python setup.py develop)
	(cd ../torchvision && python setup.py clean && python setup.py develop)
	(cd ../torchtext   && python setup.py clean && python setup.py develop)
	(cd ../torchaudio  && python setup.py clean && python setup.py develop)
	(cd ../detectron2  && python setup.py clean && python setup.py develop)
	(cd ../functorch   && python setup.py clean && python setup.py develop)
	(cd ../torchbenchmark && python install.py)
	make setup_lint

offline-autotune-cpu: develop
	rm -rf subgraphs
	python benchmarks/torchbench.py --offline-autotune -n3
	python autotune.py
	python benchmarks/torchbench.py --offline-autotune -n50

offline-autotune-gpu: develop
	rm -rf subgraphs
	python benchmarks/torchbench.py --nvfuser -d cuda --offline-autotune -n3
	python autotune.py --nvfuser
	python benchmarks/torchbench.py --nvfuser -d cuda --offline-autotune -n100

online-autotune-cpu: develop
	python benchmarks/torchbench.py --online-autotune -n50

online-autotune-gpu: develop
	python benchmarks/torchbench.py --nvfuser -d cuda --online-autotune -n100

fixed1-gpu: develop
	python benchmarks/torchbench.py --nvfuser -d cuda --speedup-fixed1 -n100

fixed2-gpu: develop
	python benchmarks/torchbench.py --nvfuser -d cuda --speedup-fixed2 -n100

baseline-cpu: develop
	 rm -f baseline_*.csv
	 python benchmarks/torchbench.py --isolate -n50 --overhead
	 python benchmarks/torchbench.py --isolate -n50 --speedup-ts
	 python benchmarks/torchbench.py --isolate -n50 --speedup-sr
	 python benchmarks/torchbench.py --isolate -n50 --speedup-onnx
	 paste -d, baseline_ts.csv baseline_sr.csv baseline_onnx.csv > baseline_all.csv

baseline-gpu: develop
	 rm -f baseline_*.csv
	 python benchmarks/torchbench.py -dcuda --isolate -n100 --overhead
	 python benchmarks/torchbench.py -dcuda --isolate -n100 --speedup-ts && mv baseline_ts.csv baseline_nnc.csv
	 python benchmarks/torchbench.py -dcuda --isolate -n100 --speedup-ts --nvfuser && mv baseline_ts.csv baseline_nvfuser.csv
	 python benchmarks/torchbench.py -dcuda --isolate -n100 --speedup-trt
	 python benchmarks/torchbench.py -dcuda --isolate -n100 --speedup-onnx
	 paste -d, baseline_nnc.csv baseline_nvfuser.csv baseline_trt.csv baseline_onnx.csv > baseline_all.csv

gpu-inductor-cudagraphs-fp32: develop
	rm -f inductor.csv baseline_cudagraphs.csv baseline_cg_nvfuser.csv baseline_cg_nnc.csv inductor_gpu_cudagraphs_fp32.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --inductor
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --backend=cudagraphs
	mv speedup_cudagraphs.csv baseline_cudagraphs.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts --nvfuser
	mv speedup_cudagraphs_ts.csv baseline_cg_nvfuser.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts
	mv speedup_cudagraphs_ts.csv baseline_cg_nnc.csv
	paste -d, inductor.csv baseline_cudagraphs.csv baseline_cg_nvfuser.csv baseline_cg_nnc.csv > inductor_gpu_cudagraphs_fp32.csv

gpu-inductor-cudagraphs-fp16: develop
	rm -f inductor.csv baseline_cudagraphs.csv baseline_cg_nvfuser.csv baseline_cg_nnc.csv inductor_gpu_cudagraphs_fp16.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float16 -n50 --inductor
	python benchmarks/torchbench.py -dcuda --inductor-settings --float16 -n50 --backend=cudagraphs
	mv speedup_cudagraphs.csv baseline_cudagraphs.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float16 -n50 --backend=cudagraphs_ts --nvfuser
	mv speedup_cudagraphs_ts.csv baseline_cg_nvfuser.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float16 -n50 --backend=cudagraphs_ts
	mv speedup_cudagraphs_ts.csv baseline_cg_nnc.csv
	paste -d, inductor.csv baseline_cudagraphs.csv baseline_cg_nvfuser.csv baseline_cg_nnc.csv > inductor_gpu_cudagraphs_fp16.csv

gpu-inductor-dynamic: develop
	rm -f inductor.csv baseline_nvfuser.csv baseline_nnc.csv inductor_gpu_dynamic.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --inductor-dynamic
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --backend=ts --nvfuser
	mv speedup_ts.csv baseline_nvfuser.csv
	python benchmarks/torchbench.py -dcuda --inductor-settings --float32 -n50 --backend=ts
	mv speedup_ts.csv baseline_nnc.csv
	paste -d, inductor.csv baseline_nvfuser.csv baseline_nnc.csv > inductor_gpu_dynamic.csv

cpu-inductor: develop
	rm -f inductor.csv speedup_ts.csv cpu_mt_inductor.csv
	python torchbench.py --inductor-settings --fast --inductor
	python torchbench.py --inductor-settings --fast --backend=ts
	paste -d, inductor.csv speedup_ts.csv > cpu_mt_inductor.csv

cpu-inductor-seq: develop
	rm -f inductor.csv speedup_ts.csv cpu_1t_inductor.csv
	taskset 1 python benchmarks/torchbench.py --inductor-settings --fast --inductor --threads=1
	taskset 1 python benchmarks/torchbench.py --inductor-settings --fast --backend=ts --threads=1
	paste -d, inductor.csv speedup_ts.csv > cpu_1t_inductor.csv





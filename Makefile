.PHONY: default develop test torchbench format lint setup clean

PY_FILES := $(wildcard *.py) $(wildcard torchdynamo/*.py) $(wildcard torchdynamo/*/*.py) \
            $(wildcard torchinductor/*.py) $(wildcard torchinductor/*/*.py)  \
            $(wildcard benchmarks/*.py) $(wildcard benchmarks/*/*.py)  \
            $(wildcard test/*.py) $(wildcard test/*/*.py)  \
            $(wildcard .circleci/*.py) $(wildcard tools/*.py)
C_FILES := $(wildcard torchdynamo/*.c torchdynamo/*.cpp)
CLANG_TIDY ?= clang-tidy-10
CLANG_FORMAT ?= clang-format-10
PIP ?= python -m pip

# versions used in CI
# Also update the "Install nightly binaries" section of the README when updating these
PYTORCH_VERSION ?= dev20221013
TRITON_VERSION ?= af76c989eb4799b015f8b288ccd8421558772e56


default: develop

develop:
	python setup.py develop

test: develop
	pytest test -o log_cli=False

torchbench: develop
	python benchmarks/torchbench.py --fast

overhead: develop
	python benchmarks/torchbench.py --overhead

format:
	isort $(PY_FILES)
	black $(PY_FILES)

lint:
	black --check --diff $(PY_FILES)
	isort --check --diff $(PY_FILES)
	flake8 $(PY_FILES)

lint-deps:
	grep -E '(black|flake8|isort|click|torch|mypy)' requirements.txt | xargs $(PIP) install

setup_lint: lint-deps

setup:
	$(PIP) install -r requirements.txt

setup_nightly:
	$(PIP) install ninja
	$(PIP) install --pre torch==1.14.0.$(PYTORCH_VERSION) --extra-index-url https://download.pytorch.org/whl/nightly/cpu
	$(PIP) install -r requirements.txt

setup_nightly_gpu:
	conda install -y -c pytorch magma-cuda116 cudatoolkit=11.6 -c conda-forge
	$(PIP) install --pre torch==1.14.0.$(PYTORCH_VERSION) \
                      torchvision==0.15.0.$(PYTORCH_VERSION) \
                      torchtext==0.14.0.$(PYTORCH_VERSION) \
                      --extra-index-url https://download.pytorch.org/whl/nightly/cu116
	$(PIP) install ninja
	$(PIP) install -U "git+https://github.com/openai/triton@$(TRITON_VERSION)#subdirectory=python"
	$(PIP) install -r requirements.txt

clean:
	python setup.py clean
	rm -rf build torchdynamo.egg-info torchdynamo/*.so __pycache__ .pytest_cache .benchmarks *.csv dist

clone-deps:
	(cd .. \
		&& (test -e pytorch || git clone --recursive https://github.com/pytorch/pytorch pytorch) \
		&& (test -e torchvision || git clone --recursive https://github.com/pytorch/vision torchvision) \
		&& (test -e torchtext || git clone --recursive https://github.com/pytorch/text torchtext) \
		&& (test -e detectron2 || git clone --recursive https://github.com/facebookresearch/detectron2) \
		&& (test -e torchbenchmark || git clone --recursive https://github.com/pytorch/benchmark torchbenchmark) \
		&& (test -e triton || git clone --recursive https://github.com/openai/triton.git) \
	)

pull-deps:
	(cd ../pytorch        && git pull && git submodule update --init --recursive)
	(cd ../torchvision    && git pull && git submodule update --init --recursive)
	(cd ../torchtext      && git pull && git submodule update --init --recursive)
	(cd ../detectron2     && git pull && git submodule update --init --recursive)
	(cd ../torchbenchmark && git pull && git submodule update --init --recursive)
	(cd ../triton         && git checkout master && git pull && git checkout $(TRITON_VERSION) && git submodule update --init --recursive)

build-deps: clone-deps
	# conda env remove --name torchdynamo
	# conda create --name torchdynamo -y python=3.8
	# conda activate torchdynamo
	conda install -y astunparse numpy scipy ninja pyyaml mkl mkl-include setuptools cmake \
        cffi typing_extensions future six requests dataclasses protobuf numba cython scikit-learn
	conda install -y -c pytorch magma-cuda116
	conda install -y -c conda-forge librosa

	make setup && $(PIP) uninstall -y torch
	(cd ../pytorch     && python setup.py clean && python setup.py develop)
	(cd ../torchvision && python setup.py clean && python setup.py develop)
	(cd ../torchtext   && python setup.py clean && python setup.py develop)
	(cd ../detectron2  && python setup.py clean && python setup.py develop)
	(cd ../torchbenchmark && python install.py --continue_on_fail)
	(cd ../triton/python && python setup.py clean && python setup.py develop)
	make setup_lint
	python setup.py develop

baseline-cpu: develop
	 rm -f baseline_*.csv
	 python benchmarks/torchbench.py -n50 --overhead
	 python benchmarks/torchbench.py -n50 --speedup-ts
	 python benchmarks/torchbench.py -n50 --speedup-sr
	 python benchmarks/torchbench.py -n50 --speedup-onnx
	 paste -d, baseline_ts.csv baseline_sr.csv baseline_onnx.csv > baseline_all.csv

baseline-gpu: develop
	 rm -f baseline_*.csv
	 python benchmarks/torchbench.py -dcuda -n100 --overhead
	 python benchmarks/torchbench.py -dcuda -n100 --speedup-ts && mv baseline_ts.csv baseline_nnc.csv
	 python benchmarks/torchbench.py -dcuda -n100 --speedup-ts --nvfuser && mv baseline_ts.csv baseline_nvfuser.csv
	 python benchmarks/torchbench.py -dcuda -n100 --speedup-trt
	 python benchmarks/torchbench.py -dcuda -n100 --speedup-onnx
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

gpu-inductor-bw-fp16: develop
	rm -f inductor.csv speedup_aot_nvfuser.csv speedup_aot_cudagraphs.csv
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float16 -n100 --backend=aot_nvfuser --nvfuser
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float16 -n100 --backend=aot_cudagraphs
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float16 -n100 --inductor
	paste -d, inductor.csv speedup_aot_nvfuser.csv speedup_aot_cudagraphs.csv > inductor_bw_fp16.csv

gpu-inductor-bw-fp32: develop
	rm -f inductor.csv speedup_aot_nvfuser.csv speedup_aot_cudagraphs.csv
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float32 -n100 --backend=aot_nvfuser --nvfuser
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float32 -n100 --backend=aot_cudagraphs
	python benchmarks/torchbench.py --training -dcuda --inductor-settings --float32 -n100 --inductor
	paste -d, inductor.csv speedup_aot_nvfuser.csv speedup_aot_cudagraphs.csv > inductor_bw_fp32.csv



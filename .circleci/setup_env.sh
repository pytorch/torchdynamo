#!/usr/bin/env bash
# based on https://github.com/pytorch/functorch/blob/main/.circleci/unittest/linux/scripts/setup_env.sh
set -x
set -e

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"
torchbench_dir="${root_dir}/torchbenchmark"

cd "${root_dir}"

if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda at ${conda_dir}\n"
    case "$(uname -s)" in
        Darwin*) os=MacOSX;;
        *) os=Linux
    esac
    wget -O miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-x86_64.sh
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environmen at ${env_dir}\n"
    conda create --prefix "${env_dir}" -y python=3.8
    conda activate "${env_dir}"
    conda install pip
    conda install -y -c pytorch magma-cuda113
    conda install -y pytorch torchvision torchaudio torchtext cudatoolkit=11.3 -c pytorch
else
    conda activate "${env_dir}"
fi

# rm -rf "${torchbench_dir}"
#
# if [ ! -d "${torchbench_dir}" ]; then
#     printf "* Installing torchbench at ${torchbench_dir}\n"
#     # conda install -y -c conda-forge git-lfs
#     # git lfs install --force --skip-repo
#     git clone --recursive git@github.com:pytorch/benchmark.git "${torchbench_dir}"
#     cd "${torchbench_dir}"
#     git lfs install --force
#     git lfs fetch
#     git lfs checkout .
#     python install.py
#     cd "${root_dir}"
# fi

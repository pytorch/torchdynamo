#!/usr/bin/env bash
# based on https://github.com/pytorch/functorch/blob/main/.circleci/unittest/linux/scripts/setup_env.sh
set -ex
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
    printf "* Creating a test environment at ${env_dir}\n"
    conda create --prefix "${env_dir}" -y python=3.8 pip
    conda activate "${env_dir}"
    make setup_nightly_gpu
else
    conda activate "${env_dir}"
fi

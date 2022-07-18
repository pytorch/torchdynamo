#!/usr/bin/env python

"""
A wrapper over the benchmark infrastructure to generate commonly used commands,
parse results and generate csv/graphs.

The script works on manually written TABLE (see below). We can add more commands
in the future.

One example usage is
-> python benchmarks/runner.py --suites=torchbench --inference
This command will generate the commands for the default compilers (see DEFAULTS
below) for inference, run them and visualize the logs.

If you want to just print the commands, you could use the following command
-> python benchmarks/runner.py --print_run_commands --suites=torchbench --inference

Similarly, if you want to just visualize the already finished logs
-> python benchmarks/runner.py --visualize_logs --suites=torchbench --inference

If you want to test float16
-> python benchmarks/runner.py --suites=torchbench --inference --dtypes=float16

"""

import argparse
import itertools
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from tabulate import tabulate

rcParams.update({"figure.autolayout": True})
plt.rc("axes", axisbelow=True)

DEFAULT_OUTPUT_DIR = "benchmark_logs"


TABLE = {
    "training": {
        "ts_nnc": "--training --speedup-ts --use-eval-mode --isolate",
        "ts_nvfuser": "--training --nvfuser --speedup-dynamo-ts --use-eval-mode --isolate",
        "aot_eager": "--training --accuracy-aot-nop --generate-aot-autograd-stats --use-eval-mode --isolate",
        "aot_nnc": "--training --accuracy-aot-ts-mincut --use-eval-mode --isolate",
        "aot_nvfuser": "--training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode --isolate",
    },
    "inference": {
        "ts_nnc": "-dcuda --isolate --speedup-ts",
        "ts_nvfuser": "-dcuda --isolate -n100 --speedup-ts --nvfuser",
        "trt": "-dcuda --isolate -n100 --speedup-trt",
        "eager_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs",
        "nnc_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts --nvfuser",
        "ts_nvfuser_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts",
        "inductor_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --inductor",
    },
}

INFERENCE_COMPILERS = tuple(TABLE["inference"].keys())
TRAINING_COMPILERS = tuple(TABLE["training"].keys())

DEFAULTS = {
    "training": ["ts_nvfuser", "aot_nvfuser"],
    "inference": ["ts_nvfuser_cudagraphs", "inductor_cudagraphs"],
    "dtypes": [
        "float32",
    ],
    "suites": ["torchbench", "huggingface"],
    "devices": [
        "cuda",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", action="append", help="cpu or cuda")
    parser.add_argument("--dtypes", action="append", help="float16/float32/amp")
    parser.add_argument("--suites", action="append", help="huggingface/torchbench/timm")
    parser.add_argument(
        "--compilers",
        action="append",
        help=f"For --inference, options are {INFERENCE_COMPILERS}. For --training, options are {TRAINING_COMPILERS}",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Just runs one model. Helps in debugging"
    )
    parser.add_argument(
        "--output-dir", help="Choose the output directory to save the logs"
    )

    # Choose either generation of commands, pretty parsing or e2e runs
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--print_run_commands",
        action="store_true",
        help="Generate commands and saves them to run.sh",
    )
    group.add_argument(
        "--visualize_logs",
        action="store_true",
        help="Pretty print the log files and draw graphs",
    )
    group.add_argument(
        "--run",
        action="store_true",
        default=True,
        help="Generate commands, run and parses the files",
    )

    # Choose either inference or training
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument(
        "--inference", action="store_true", help="Only run inference related tasks"
    )
    group_mode.add_argument(
        "--training", action="store_true", help="Only run training related tasks"
    )

    args = parser.parse_args()
    return args


def generate_commands(args, dtypes, suites, devices, compilers, output_dir):
    mode = "inference" if args.inference else "training"
    with open("run.sh", "w") as runfile:
        lines = []

        lines.append("# Setup the output directory")
        lines.append(f"rm -rf {output_dir}")
        lines.append(f"mkdir {output_dir}")
        lines.append("")

        for iter in itertools.product(suites, devices, dtypes):
            suite, device, dtype = iter
            lines.append(
                f"# Commands for {suite} for device={device}, dtype={dtype} for {mode}"
            )

            info = TABLE[mode]
            for compiler in compilers:
                base_cmd = info[compiler]
                output_filename = (
                    f"{output_dir}/{compiler}_{suite}_{dtype}_{mode}_{device}.csv"
                )
                cmd = f"python benchmarks/{suite}.py --{dtype} -d{device} --output={output_filename} {base_cmd}"
                if args.quick:
                    if suite == "torchbench":
                        cmd = f"{cmd} --only=resnet18"
                    elif suite == "huggingface":
                        cmd = f"{cmd} --only=BertForPreTraining_P1_bert"
                    else:
                        raise NotImplementedError(
                            f"Quick not implemented for {suite}.py"
                        )
                lines.append(cmd)
            lines.append("")
        runfile.writelines([line + "\n" for line in lines])


def pp_dataframe(df, title, output_dir):
    # Pretty print
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex="never"))

    # Save to csv, can be copy pasted in google sheets
    df.to_csv(f"{output_dir}/{title}.csv", index=False)

    # Graph
    labels = df.columns.values.tolist()
    labels = labels[2:]
    df.plot(
        x="name",
        y=labels,
        kind="bar",
        title=title,
        ylabel="Speedup over eager",
        xlabel="",
        grid=True,
        figsize=(max(len(df.index) / 4, 5), 10),
        edgecolor="black",
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title}.png")


def read_csv(output_filename):
    has_header = False
    n_cols = 3
    with open(output_filename, "r") as f:
        line = f.readline()
        if "dev" in line:
            has_header = True
            n_cols = len(line.rstrip().split())

    if has_header:
        return pd.read_csv(output_filename)
    else:
        assert n_cols == 3
        return pd.read_csv(
            output_filename, names=["dev", "name", "speedup"], header=None
        )


def parse_logs(args, dtypes, suites, devices, compilers, output_dir):
    mode = "inference" if args.inference else "training"
    for iter in itertools.product(suites, devices, dtypes):
        suite, device, dtype = iter
        frames = []
        best_compiler = compilers[-1]
        # Collect results from all the files
        for compiler in compilers:
            output_filename = (
                f"{output_dir}/{compiler}_{suite}_{dtype}_{mode}_{device}.csv"
            )

            df = read_csv(output_filename)
            df.rename(
                columns={"speedup": compiler, "ts": compiler, "ofi": f"ofi_{compiler}"},
                inplace=True,
            )
            frames.append(df)

        # Merge the results
        if len(compilers) == 1:
            df = frames[0]
        else:
            df = pd.merge(*frames, on=["dev", "name"])

        # Pretty print and also write to a bargraph
        title = f"{suite}_{dtype}_{mode}_{device}"
        pp_dataframe(df, title, output_dir)

        # Sort the dataframe and pretty print
        sorted_df = df.sort_values(by=best_compiler, ascending=False)
        pp_dataframe(sorted_df, f"sorted_{title}", output_dir)


if __name__ == "__main__":
    args = parse_args()

    def extract(key):
        return DEFAULTS[key] if getattr(args, key, None) is None else getattr(args, key)

    dtypes = extract("dtypes")
    suites = extract("suites")
    devices = extract("devices")

    if args.inference:
        compilers = DEFAULTS["inference"] if args.compilers is None else args.compilers
    else:  # args.training
        compilers = DEFAULTS["training"] if args.compilers is None else args.compilers

    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    if args.print_run_commands:
        generate_commands(args, dtypes, suites, devices, compilers, output_dir)
    elif args.visualize_logs:
        parse_logs(args, dtypes, suites, devices, compilers, output_dir)
    elif args.run:
        generate_commands(args, dtypes, suites, devices, compilers, output_dir)
        # TODO - Do we need to worry about segfaults
        try:
            os.system("bash run.sh")
        except Exception as e:
            print(
                "Running commands failed. Please run manually (bash run.sh) and inspect the errors."
            )
            raise e
        parse_logs(args, dtypes, suites, devices, compilers, output_dir)

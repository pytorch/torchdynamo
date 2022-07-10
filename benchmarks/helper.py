#!/usr/bin/env python

"""
A wrapper over the benchmark infrastructure to generate commonly use commands,
parse results and generate csv/graphs.

The script works on manually written TABLE (see below). We can add more commands
in the future.

One example usage is
-> python benchmarks/helper.py --e2e --modes=training --devices=cuda --suites=torchbench --dtypes=float32 --nightly

where
    e2e - generates the commands, runs the benchmarks, parse the results (other options - commands/parse)
    nightly - Looks at the nightly field in the training/inferece field of TABLE to choose most relevant compilers.
    models - either training or inference
    suites - benchmark suite

In case, you just need to get the commands, you can replace e2e with commands
-> python benchmarks/helper.py --commands --modes=training --devices=cuda --suites=torchbench --dtypes=float32 --nightly

The commands are written in file run.sh.

In case, you just want to parse the logs once you have run the commands manually, you can use this command
-> python benchmarks/helper.py --parse --modes=training --devices=cuda --suites=torchbench --dtypes=float32 --nightly

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
    "suites": ["torchbench", "huggingface"],
    "dtypes": ["float32", "float16", "amp"],
    "devices": ["cuda", "cpu"],
    "modes": ["inference", "training"],
    # Dict of name to base_command. nightly is a special field to tell which compilers are relevant.
    "training": {
        "ts_nnc": "--training --speedup-ts --use-eval-mode --isolate",
        "ts_nvfuser": "--training --nvfuser --speedup-ts --use-eval-mode --isolate",
        "aot_eager": "--training --accuracy-aot-nop --generate-aot-autograd-stats --use-eval-mode --isolate",
        "aot_nnc": "--training --accuracy-aot-ts-mincut --use-eval-mode --isolate",
        "aot_nvfuser": "--training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode --isolate",
        "nightly": ["ts_nvfuser", "aot_nvfuser"],
    },
    # Dict of name to base_command. nightly is a special field to tell which compilers are relevant.
    "inference": {
        "ts_nnc": "-dcuda --isolate --speedup-ts",
        "ts_nvfuser": "-dcuda --isolate -n100 --speedup-ts --nvfuser",
        "trt": "-dcuda --isolate -n100 --speedup-trt",
        "eager_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs",
        "nnc_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts --nvfuser",
        "nvfuser_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --backend=cudagraphs_ts",
        "inductor_cudagraphs": "-dcuda --inductor-settings --float32 -n50 --inductor",
        "nightly": ["nvfuser_cudagraphs", "inductor_cudagraphs"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument("--devices", action="append", help="cpu or cuda")
    parser.add_argument("--modes", action="append", help="inference or training")
    parser.add_argument("--dtypes", action="append", help="float16/float32/amp")
    parser.add_argument("--suites", action="append", help="huggingface/torchbench")
    parser.add_argument("--quick", action="store_true", help="Just runs one model")
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="Use only compilers mentioned in nightly field",
    )
    parser.add_argument(
        "--output-dir", help="Choose the output directory to save the logs"
    )

    # Choose either generation of commands, pretty parsing or e2e runs
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--commands", action="store_true", help="Generate commands")
    group.add_argument("--parse", action="store_true", help="Parse the files")
    group.add_argument(
        "--e2e", action="store_true", help="Generate commands, run and parse the files"
    )

    args = parser.parse_args()
    return args


def generate_commands(args, dtypes, suites, modes, devices, output_dir):
    with open("run.sh", "w") as runfile:
        lines = []

        lines.append("# Setup the output directory")
        lines.append(f"rm -rf {output_dir}")
        lines.append(f"mkdir {output_dir}")
        lines.append("")

        for iter in itertools.product(modes, suites, devices, dtypes):
            mode, suite, device, dtype = iter
            lines.append(
                f"# Commands for {suite} for device={device}, dtype={dtype} for {mode}"
            )

            info = TABLE[mode]
            compilers = list(info.keys())
            compilers.remove("nightly")
            if args.nightly:
                compilers = info["nightly"]
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
        ylabel="Speedeup over eager",
        xlabel="",
        grid=True,
        figsize=(max(len(df.index) / 3, 10), 10),
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{title}.png")


def parse_logs(args, dtypes, suites, modes, devices, output_dir):
    for iter in itertools.product(modes, suites, devices, dtypes):
        mode, suite, device, dtype = iter
        info = TABLE[mode]
        compilers = list(info.keys())
        compilers.remove("nightly")
        if args.nightly:
            compilers = info["nightly"]
        frames = []
        best_compiler = compilers[-1]
        # Collect results from all the files
        for compiler in compilers:
            output_filename = (
                f"{output_dir}/{compiler}_{suite}_{dtype}_{mode}_{device}.csv"
            )
            df = pd.read_csv(output_filename)
            df.rename(
                columns={"speedup": compiler, "ts": compiler, "ofi": f"ofi_{compiler}"},
                inplace=True,
            )
            frames.append(df)

        # Merge the results
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
        value = getattr(args, key, None)
        return value if value is not None else TABLE[key]

    dtypes = extract("dtypes")
    suites = extract("suites")
    modes = extract("modes")
    devices = extract("devices")

    output_dir = args.output_dir if args.output_dir is not None else DEFAULT_OUTPUT_DIR

    if args.commands:
        generate_commands(args, dtypes, suites, modes, devices, output_dir)
    elif args.parse:
        parse_logs(args, dtypes, suites, modes, devices, output_dir)
    elif args.e2e:
        generate_commands(args, dtypes, suites, modes, devices, output_dir)
        os.system("bash run.sh")
        parse_logs(args, dtypes, suites, modes, devices, output_dir)

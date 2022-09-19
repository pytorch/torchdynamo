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
import importlib
import io
import itertools
import os
from collections import defaultdict
from os.path import abspath
from os.path import exists

import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib import rcParams
from numpy.core.fromnumeric import mean
from scipy.stats import gmean
from tabulate import tabulate

import torchdynamo

rcParams.update({"figure.autolayout": True})
plt.rc("axes", axisbelow=True)

DEFAULT_OUTPUT_DIR = "benchmark_logs"


TABLE = {
    "training": {
        "ts_nnc": "--training --speedup-ts --use-eval-mode ",
        "ts_nvfuser": "--training --nvfuser --speedup-dynamo-ts --use-eval-mode ",
        "eager": "--training --backend=eager --use-eval-mode",
        "aot_eager": "--training --accuracy-aot-nop --generate-aot-autograd-stats --use-eval-mode ",
        "aot_cudagraphs": "--training --backend=aot_cudagraphs --use-eval-mode ",
        "aot_nnc": "--training --accuracy-aot-ts-mincut --use-eval-mode ",
        "aot_nvfuser": "--training --nvfuser --accuracy-aot-ts-mincut --use-eval-mode ",
        "inductor_cudagraphs": "--training --inductor --use-eval-mode",
    },
    "inference": {
        "ts_nnc": "--speedup-ts",
        "ts_nvfuser": "-n100 --speedup-ts --nvfuser",
        "trt": "-n100 --speedup-trt",
        "ts_nvfuser_cudagraphs": "--inductor-settings --float32 -n50 --backend=cudagraphs_ts",
        "inductor_cudagraphs": "--inductor-settings --float32 -n50 --inductor",
    },
    "profile_compiler": {
        "pytorch": "--training --profile-backend=pytorch",
        "eager": "--training --profile-backend=eager",
        "ts_nvfuser": "--training --profile-backend=nvfuser",
        "aot_eager": "--training --profile-backend=aot_eager",
        "aot_nvfuser": "--training --profile-backend=aot_nvfuser",
        "inductor_cudagraphs": "--training --profile-backend=inductor",
    },
}

INFERENCE_COMPILERS = tuple(TABLE["inference"].keys())
TRAINING_COMPILERS = tuple(TABLE["training"].keys())

DEFAULTS = {
    "training": [
        "eager",
        "ts_nvfuser",
        "aot_eager",
        "aot_cudagraphs",
        "aot_nvfuser",
        "inductor_cudagraphs",
    ],
    "inference": ["ts_nvfuser_cudagraphs", "inductor_cudagraphs"],
    "profile_compiler": [
        "pytorch",
        "eager",
        "aot_eager",
        "aot_nvfuser",
        "inductor_cudagraphs",
    ],
    "dtypes": [
        "float32",
    ],
    "suites": ["torchbench", "huggingface", "timm_models"],
    "devices": [
        "cuda",
    ],
    "quick": {
        "torchbench": ["resnet18", "resnet50"],
        "huggingface": ["AlbertForMaskedLM", "BertForMaskedLM"],
        "timm_models": ["resnest101e", "mobilenetv2_100"],
    },
}


def percentage(part, whole, decimals=2):
    return round(100 * float(part) / float(whole), decimals)


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

    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
        help="Log operator inputs",
    )

    # Choose either inference or training
    group_mode = parser.add_mutually_exclusive_group(required=True)
    group_mode.add_argument(
        "--inference", action="store_true", help="Only run inference related tasks"
    )
    group_mode.add_argument(
        "--training", action="store_true", help="Only run training related tasks"
    )
    group_mode.add_argument(
        "--profile_compiler",
        action="store_true",
        help="Runs profile_compiler experiment",
    )

    args = parser.parse_args()
    return args


def get_mode(args):
    if args.inference:
        return "inference"
    elif args.training:
        return "training"
    else:
        assert args.profile_compiler
        return "profile_compiler"


def get_skip_tests(suite):
    """
    Generate -x seperated string to skip the unusual setup training tests
    """
    skip_tests = set()
    original_dir = abspath(os.getcwd())
    module = importlib.import_module(suite)
    os.chdir(original_dir)

    if hasattr(module, "SKIP"):
        skip_tests.update(module.SKIP)
    if hasattr(module, "SKIP_TRAIN"):
        skip_tests.update(module.SKIP_TRAIN)

    skip_tests = map(lambda name: f"-x {name}", skip_tests)
    skip_str = " ".join(skip_tests)
    return skip_str


def generate_commands(args, dtypes, suites, devices, compilers, output_dir):
    mode = get_mode(args)
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
                cmd = f"python benchmarks/{suite}.py --{dtype} -d{device} --no-skip --output={output_filename} --quiet"
                cmd = f"{cmd} {base_cmd}"
                if args.profile_compiler:
                    cmd = f"{cmd} --raise-on-assertion-error --raise-on-backend-error"

                skip_tests_str = get_skip_tests(suite)
                cmd = f"{cmd} {skip_tests_str}"

                if args.log_operator_inputs:
                    cmd = f"{cmd} --log-operator-inputs"

                if args.quick:
                    for name in DEFAULTS["quick"][suite]:
                        new_cmd = f"{cmd} --only={name}"
                        lines.append(new_cmd)
                else:
                    lines.append(cmd)
            lines.append("")
        runfile.writelines([line + "\n" for line in lines])


def generate_dropdown_comment(title, body):
    str_io = io.StringIO()
    str_io.write(f"{title}\n")
    str_io.write("<details>\n")
    str_io.write("<summary>see more</summary>\n")
    str_io.write(f"{body}")
    str_io.write("\n")
    str_io.write("</details>\n\n")
    return str_io.getvalue()


def build_summary():
    import git

    out_io = io.StringIO()

    def print_commit_hash(path, name):
        if exists(path):
            repo = git.Repo(path, search_parent_directories=True)
            sha = repo.head.object.hexsha
            out_io.write(f"{name} commit: {sha}\n")
        else:
            out_io.write(f"{name} Absent\n")

    def env_var(name):
        out_io.write(f"{name} = {os.environ[name]}\n")

    out_io.write("## Commit hashes ##\n")
    print_commit_hash(".", "torchdynamo")
    print_commit_hash("../pytorch", "pytorch")
    print_commit_hash("../functorch", "functorch")
    print_commit_hash("../torchbenchmark", "torchbench")

    out_io.write("\n")
    out_io.write("## TorchDynamo config flags ##\n")
    for key in dir(torchdynamo.config):
        val = getattr(torchdynamo.config, key)
        if not key.startswith("__") and isinstance(val, bool):
            out_io.write(f"torchdynamo.config.{key} = {val}\n")

    out_io.write("\n")
    out_io.write("## Torch version ##\n")
    out_io.write(f"torch: {torch.__version__}\n")

    out_io.write("\n")
    out_io.write("## Environment variables ##\n")
    env_var("TORCH_CUDA_ARCH_LIST")
    env_var("CUDA_HOME")
    env_var("USE_LLVM")

    out_io.write("\n")
    out_io.write("## GPU details ##\n")
    out_io.write(f"CUDNN VERSION: {torch.backends.cudnn.version()}\n")
    out_io.write(f"Number CUDA Devices: {torch.cuda.device_count()}\n")
    out_io.write(f"Device Name: {torch.cuda.get_device_name(0)}\n")
    out_io.write(
        f"Device Memory [GB]: {torch.cuda.get_device_properties(0).total_memory/1e9}\n"
    )

    title = "## Build Summary"
    comment = generate_dropdown_comment(title, out_io.getvalue())
    with open(f"{output_dir}/gh_build_summary.txt", "w") as gh_fh:
        gh_fh.write(comment)


class Parser:
    def __init__(self, suites, devices, dtypes, compilers, mode, output_dir):
        self.suites = suites
        self.devices = devices
        self.dtypes = dtypes
        self.compilers = compilers
        self.output_dir = output_dir
        self.mode = mode

    def has_header(self, output_filename):
        header_present = False
        with open(output_filename, "r") as f:
            line = f.readline()
            if "dev" in line:
                header_present = True
        return header_present

    def gen_github_comment(self):
        comment = self.prettyprint()
        print(comment)
        with open(f"{self.output_dir}/gh_{self.mode}.txt", "w") as gh_fh:
            gh_fh.write(comment)


class ParseCompilerProfileLogs(Parser):
    def __init__(self, suites, devices, dtypes, compilers, mode, output_dir):
        super().__init__(suites, devices, dtypes, compilers, mode, output_dir)
        self.parsed_frames = {}
        self.metrics = ["time", "memory", "graphs"]
        self.title = {
            "time": "Compilation Latency",
            "memory": "Peak Memory",
            "graphs": "Number of graphs",
        }
        self.threshold = 50
        self.units = {
            "time": "seconds",
            "memory": "GB",
            "graphs": "graphs",
        }
        self.parse()

    def read_csv(self, output_filename):
        assert self.has_header(output_filename)
        return pd.read_csv(output_filename)

    def parse(self):
        for metric in self.metrics:
            self.parsed_frames[metric] = self.extract_df(metric)

    def extract_df(self, metric):
        # frames = collections.defaultdict()
        frames_per_suite = []
        for iter in itertools.product(self.suites, self.devices, self.dtypes):
            suite, device, dtype = iter
            # Collect results from all the files
            frames = []
            for compiler in self.compilers:
                output_filename = f"{self.output_dir}/{compiler}_{suite}_{dtype}_{self.mode}_{device}.csv"

                df = self.read_csv(output_filename)
                df.insert(1, "suite", suite)
                batch_size_idx = df.columns.to_list().index("batch_size")
                common_columns = df.columns.to_list()[: batch_size_idx + 1]
                subset_df = df[df.columns[0 : batch_size_idx + 1]]
                subset_df.insert(batch_size_idx + 1, compiler, df[metric])
                frames.append(subset_df)

            if len(frames) == 1:
                df = frames[0]
            else:
                # Merge data frames
                df = pd.merge(frames[0], frames[1], on=common_columns)
                for idx in range(2, len(frames)):
                    df = pd.merge(df, frames[idx], on=common_columns)

            baseline = df["pytorch"]
            for compiler in self.compilers:
                df[compiler] = df[compiler] - baseline
            frames_per_suite.append(df)

        # Concat data frames
        if len(frames_per_suite) == 1:
            df = frames_per_suite[0]
        else:
            df = pd.concat(frames_per_suite)

        # Sort in descending order
        # df = df.sort_values(by=list(reversed(self.compilers)), ascending=False)
        df = df.sort_values(by=self.compilers[-2], ascending=False)
        df = df.round(3)

        # For graph breaks, just print one column
        if metric == "graphs":
            batch_size_idx = df.columns.to_list().index("batch_size")
            common_columns = df.columns.to_list()[: batch_size_idx + 1]
            subset_df = df[df.columns[0 : batch_size_idx + 1]]
            subset_df.insert(batch_size_idx + 1, "graphs", df["eager"])
            df = subset_df
        return df

    def prepare_message_for_metric(self, metric):
        pd.options.display.float_format = "{:,.2f}".format
        title = f"## {self.title[metric]} ##"
        df = self.parsed_frames[metric]
        df = df.head(self.threshold)
        df = df.drop("dev", axis=1)
        tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write(f"dtype={self.dtypes[0]}, unit={self.units[metric]}\n")
        str_io.write("~~~\n")
        str_io.write(f"{tabform}\n")
        str_io.write("~~~\n")
        body = str_io.getvalue()
        comment = generate_dropdown_comment(title, body)
        return comment

    def prettyprint(self):
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write("# Compilation Profile #\n")
        str_io.write(
            f"The tables show the worst {self.threshold} models for different metrics"
        )
        str_io.write("\n")
        for metric in self.metrics:
            str_io.write(self.prepare_message_for_metric(metric))
        str_io.write("\n")
        return str_io.getvalue()


class ParsePerformanceLogs(Parser):
    def __init__(self, suites, devices, dtypes, compilers, mode, output_dir):
        super().__init__(suites, devices, dtypes, compilers, mode, output_dir)
        self.parsed_frames = defaultdict(lambda: defaultdict(None))
        self.metrics = ["speedup", "start_latency", "peak_memory"]
        self.bottom_k = 50
        self.parse()

    def plot_graph(self, df, title):
        labels = df.columns.values.tolist()
        labels = labels[3:]
        df.plot(
            x="name",
            y=labels,
            kind="bar",
            width=0.65,
            title=title,
            ylabel="Speedup over eager",
            xlabel="",
            grid=True,
            figsize=(max(len(df.index) / 4, 5), 10),
            edgecolor="black",
        )
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{title}.png")

    def read_csv(self, output_filename):
        if self.has_header(output_filename):
            return pd.read_csv(output_filename)
        else:
            return pd.read_csv(
                output_filename,
                names=[
                    "dev",
                    "name",
                    "batch_size",
                    "speedup",
                    "start_latency",
                    "peak_memory",
                ],
                header=None,
            )

    def parse(self):
        for metric in self.metrics:
            self.extract_df(metric)
        self.generate_executive_summary()
        for suite in self.suites:
            self.plot_graph(
                self.parsed_frames[suite]["speedup"], f"{suite}_{self.dtypes[0]}"
            )

    def clean_batch_sizes(self, frames):
        # Clean up batch sizes when its 0
        if len(frames) == 1:
            return frames
        batch_sizes = frames[0]["batch_size"].to_list()
        for frame in frames[1:]:
            frame_batch_sizes = frame["batch_size"].to_list()
            for idx, (batch_a, batch_b) in enumerate(
                zip(batch_sizes, frame_batch_sizes)
            ):
                assert batch_a == batch_b or batch_a == 0 or batch_b == 0, print(
                    f"a={batch_a}, b={batch_b}"
                )
                batch_sizes[idx] = max(batch_a, batch_b)
        for frame in frames:
            frame["batch_size"] = batch_sizes
        return frames

    def extract_df(self, metric):
        for iter in itertools.product(self.suites, self.devices, self.dtypes):
            suite, device, dtype = iter
            frames = []
            for compiler in self.compilers:
                output_filename = f"{self.output_dir}/{compiler}_{suite}_{dtype}_{self.mode}_{device}.csv"
                df = self.read_csv(output_filename)
                df = df[["dev", "name", "batch_size", metric]]
                df.rename(columns={metric: compiler}, inplace=True)
                df["batch_size"] = df["batch_size"].astype(int)
                frames.append(df)

            # Merge the results
            frames = self.clean_batch_sizes(frames)
            if len(self.compilers) == 1:
                df = frames[0]
            else:
                # Merge data frames
                df = pd.merge(frames[0], frames[1], on=["dev", "name", "batch_size"])
                for idx in range(2, len(frames)):
                    df = pd.merge(df, frames[idx], on=["dev", "name", "batch_size"])

            df = df.sort_values(by=list(reversed(self.compilers)), ascending=False)
            self.parsed_frames[suite][metric] = df

    def comp_time(self, compiler, df):
        df = df.sort_values(by=compiler, ascending=False)[compiler][: self.bottom_k]
        return f"{mean(df):.2f}"

    def geomean(self, compiler, df):
        cleaned_df = df[compiler][df[compiler] > 0].clip(1)
        if cleaned_df.empty:
            return "0.0x"
        return f"{gmean(cleaned_df):.2f}x"

    def passrate(self, compiler, df):
        total = len(df.index)
        passing = df[df[compiler] > 0.0][compiler].count()
        perc = int(percentage(passing, total, decimals=0))
        return f"{perc}%, {passing}/{total}"

    def exec_summary_df(self, fn, metric):
        """
        Generate a table with passrate and geomean perf
        """
        cols = {}
        cols["Compiler"] = self.compilers
        for suite in self.suites:
            df = self.parsed_frames[suite][metric]
            # speedups = [self.geomean(compiler, df) for compiler in self.compilers]
            speedups = [fn(compiler, df) for compiler in self.compilers]
            col = pd.Series(data=speedups, index=self.compilers)
            cols[suite] = col
        df = pd.DataFrame(cols)
        df = df.fillna(0)
        df.to_csv(os.path.join(self.output_dir, f"{fn.__name__}.csv"))
        return df

    def exec_summary_text(self, caption, fn, metric):
        df = self.exec_summary_df(fn, metric)
        tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")

        str_io = io.StringIO()
        str_io.write(f"{caption}")
        str_io.write("~~~\n")
        str_io.write(f"{tabform}\n")
        str_io.write("~~~\n")
        return str_io.getvalue()

    def generate_executive_summary(self):
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write("## Executive Summary ##\n")
        description = (
            "This table shows the accuracy and performance numbers for different backends "
            "across three benchmark suites - torchbench, huggingface and timm. We run "
            "these experiments on A100 GPUs. Each experiment runs one iteration of forward "
            "and backward pass. For accuracy, we check the numerical correctness of forward "
            "pass outputs and gradients by comparing with native pytorch. We measure speedup "
            "by normalizing against the performance of native pytorch. We also report compilation "
            f"time metric which is mean of worst {self.bottom_k} compilation models.\n\n"
            "Caveats\n"
            "1) Batch size has been reduced to workaround OOM errors. Work is in progress to "
            "reduce peak memory footprint.\n"
            "2) Experiments do not cover dynamic shapes.\n"
            "3) Experimental setup does not have optimizer.\n\n"
        )
        str_io.write(description)

        speedup_caption = "Geometric mean speedup\n"
        speedup_summary = self.exec_summary_text(
            speedup_caption, self.geomean, "speedup"
        )

        passrate_caption = "Passrate\n"
        passrate_summary = self.exec_summary_text(
            passrate_caption, self.passrate, "speedup"
        )

        comp_time_caption = (
            f"Mean compilation time (seconds) for worst {self.bottom_k} models\n"
        )
        comp_time_summary = self.exec_summary_text(
            comp_time_caption, self.comp_time, "start_latency"
        )

        str_io.write(passrate_summary)
        str_io.write(speedup_summary)
        str_io.write(comp_time_summary)
        self.executive_summary = str_io.getvalue()

    def prepare_message(self, suite):
        title = f"## {suite} suite with {self.dtypes[0]} precision ##"
        body = ""
        for metric in ["speedup", "start_latency"]:
            df = self.parsed_frames[suite][metric]
            df = df.drop("dev", axis=1)
            tabform = tabulate(df, headers="keys", tablefmt="pretty", showindex="never")
            str_io = io.StringIO()
            str_io.write("\n")
            if metric == "speedup":
                str_io.write("Performance speedup\n")
            elif metric == "start_latency":
                str_io.write("Compilation latency\n")
            str_io.write("~~~\n")
            str_io.write(f"{tabform}\n")
            str_io.write("~~~\n")
            body += str_io.getvalue()

        comment = generate_dropdown_comment(title, body)
        return comment

    def prettyprint(self):
        str_io = io.StringIO()
        str_io.write("\n")
        str_io.write(f"# Performance Dashboard for {self.dtypes[0]} precision ##\n")
        str_io.write("\n")
        str_io.write(self.executive_summary)
        for suite in self.suites:
            str_io.write(self.prepare_message(suite))
        str_io.write("\n")
        return str_io.getvalue()


def parse_logs(args, dtypes, suites, devices, compilers, output_dir):
    mode = get_mode(args)
    build_summary()

    parser_class = ParsePerformanceLogs
    if args.profile_compiler:
        parser_class = ParseCompilerProfileLogs

    parser = parser_class(suites, devices, dtypes, compilers, mode, output_dir)
    parser.gen_github_comment()
    return


if __name__ == "__main__":
    args = parse_args()

    def extract(key):
        return DEFAULTS[key] if getattr(args, key, None) is None else getattr(args, key)

    dtypes = extract("dtypes")
    suites = extract("suites")
    devices = extract("devices")

    if args.inference:
        compilers = DEFAULTS["inference"] if args.compilers is None else args.compilers
    elif args.training:  # args.training
        compilers = DEFAULTS["training"] if args.compilers is None else args.compilers
    else:
        assert args.profile_compiler
        compilers = (
            DEFAULTS["profile_compiler"] if args.compilers is None else args.compilers
        )

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
        if not args.log_operator_inputs:
            parse_logs(args, dtypes, suites, devices, compilers, output_dir)

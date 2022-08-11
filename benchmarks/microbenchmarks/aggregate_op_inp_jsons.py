import json
import os
from collections import Counter
from collections import defaultdict
from os import listdir
from os.path import isfile

import click
from operator_inp_utils import OperatorInputsMode


def get_files_for_directory(directory):
    return [f for f in listdir(directory) if isfile(os.path.join(directory, f))]


def aggregate_file(file_name, db):
    file = open(file_name)
    for op, op_inputs in json.load(file).items():
        for inps, counter in op_inputs.items():
            db[op][inps] += counter


def aggregate_files(directory, output_filename):
    aggregate_db = defaultdict(Counter)
    files = get_files_for_directory(directory)
    for f in files:
        aggregate_file(f"{directory}/{f}", aggregate_db)

    OperatorInputsMode(aggregate_db).log_to_file(output_filename)


@click.command()
@click.option("--directory", help="Directory to aggregate from.")
@click.option("--output_filename", default=None, help="full path of aggregated output")
def aggregate(directory, output_filename):
    aggregate_files(directory, output_filename)


if __name__ == "__main__":
    aggregate()

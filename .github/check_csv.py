import argparse
import sys
import textwrap

import pandas as pd


def check_csv(filename):
    """
    Basic accuracy checking. If a model fails, the speedup will be 0.
    """

    actual = pd.read_csv(filename)

    failed = []
    for model_name in actual["name"]:
        speedup = float(actual.loc[actual["name"] == model_name]["speedup"])
        status = "PASS"
        if speedup == 0:
            status = "FAIL"
            failed.append(model_name)

        print(f"{model_name:34} {status}")

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models failed
                    {' '.join(failed)}
                """
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    args = parser.parse_args()
    check_csv(args.file)

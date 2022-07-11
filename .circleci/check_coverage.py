import sys
import textwrap

import pandas as pd


def main():
    expected = pd.read_csv(".circleci/expected_coverage.csv")
    actual = pd.read_csv("coverage.csv")
    failed = []

    for name in expected["name"]:
        expected_ops = int(expected.loc[expected["name"] == name]["captured_ops"])
        try:
            actual_ops = int(actual.loc[actual["name"] == name]["captured_ops"])
        except TypeError:
            print(f"{name:34} MISSING")
            failed.append(name)
            continue
        if actual_ops >= max(1, min(expected_ops - 10, expected_ops * 0.9)):
            status = "PASS"
        else:
            status = "FAIL"
            failed.append(name)
        print(
            f"{name:34} actual_ops={actual_ops:4} expected_ops={expected_ops:4} {status}"
        )

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models below expected coverage:
                    {' '.join(failed)}

                If this coverage drop is expected, then you can update targets
                by downloading `coverage.csv` from the artifacts tab in CircleCI
                and replacing `.circleci/expected_coverage.csv`.
                """
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

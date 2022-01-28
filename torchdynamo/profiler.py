import dataclasses
from typing import List, Any

import torch


@dataclasses.dataclass
class ProfileMetrics:
    microseconds: float = 0.0
    operators: int = 0
    fusions: int = 0
    graphs: int = 0

    def __iadd__(self, other: "ProfileMetrics"):
        self.microseconds += other.microseconds
        self.operators += other.operators
        self.fusions += other.fusions
        return self

    def __add__(self, other: "ProfileMetrics"):
        assert isinstance(other, ProfileMetrics)
        return ProfileMetrics(
            self.microseconds + other.microseconds,
            self.operators + other.operators,
            self.fusions + other.fusions,
        )

    def __truediv__(self, other):
        if isinstance(other, int):
            other = ProfileMetrics(other, other, other)
        return ProfileMetrics(
            self.microseconds / max(1, other.microseconds),
            self.operators / max(1, other.operators),
            self.fusions / max(1, other.fusions),
        )

    def __str__(self):
        return f"{self.operators:4.0%} ops {self.microseconds:4.0%} time"

    def tocsv(self):
        return [self.operators, self.microseconds]


class ProfileResult:
    def __init__(self, captured=None, total=None):
        self.captured: ProfileMetrics = captured or ProfileMetrics()
        self.total: ProfileMetrics = total or ProfileMetrics()

    def __iadd__(self, other: ProfileMetrics):
        self.captured += other.captured
        self.total += other.total
        return self

    def percent(self):
        return self.captured / self.total

    def __str__(self):
        return (
            f"{self.captured.graphs:2} graphs {self.captured.operators:4}/{self.total.operators:4} ops, "
            + str(self.percent())
        )

    def tocsv(self):
        return self.percent().tocsv() + [
            self.captured.operators,
            self.total.operators - self.captured.operators,
            self.captured.graphs,
        ]


class Profiler:
    def __init__(self):
        self.prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        )

    def results(self):
        captured_regions = 0
        captured_ops = 0
        captured_microseconds = 0
        total_ops = 0
        total_microseconds = 0

        last_op_end_time = -1
        captured_region_end_time = -1
        events = list(sorted(self.prof.events(), key=lambda x: x.time_range.start))
        for e in events:
            if e.name == "TORCHDYNAMO":
                captured_region_end_time = e.time_range.end
                captured_regions += 1
            elif e.time_range.start >= last_op_end_time:
                last_op_end_time = e.time_range.end
                if e.time_range.end <= captured_region_end_time:
                    captured_ops += 1
                    captured_microseconds += e.time_range.elapsed_us()
                total_ops += 1
                total_microseconds += e.time_range.elapsed_us()
            else:
                pass  # ops recursively called from other ops (ignored)

        return ProfileResult(
            captured=ProfileMetrics(
                microseconds=captured_microseconds,
                operators=captured_ops,
                fusions=captured_ops - captured_regions,
                graphs=captured_regions,
            ),
            total=ProfileMetrics(
                microseconds=total_microseconds,
                operators=total_ops,
                fusions=total_ops - 1,
            ),
        )


def fx_insert_profiling(gm: torch.fx.GraphModule, example_inputs: List[Any]):
    def _wrapped(*args, **kwargs):
        with torch.profiler.record_function("TORCHDYNAMO"):
            return gm.forward(*args, **kwargs)

    return _wrapped

import dataclasses
import os
import traceback

from torchdynamo.utils import counters


class InternalTorchDynamoError(RuntimeError):
    pass


class RestartAnalysis(RuntimeError):
    pass


class SkipFrame(RuntimeError):
    pass


class TorchRuntimeError(RuntimeError):
    pass


@dataclasses.dataclass
class FakeTensorError(RuntimeError):
    reason: str


class BackendCompilerFailed(RuntimeError):
    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, "__name__", "?")
        self.inner_exception = inner_exception
        super().__init__(
            f"{self.backend_name} raised {type(inner_exception).__name__}: {inner_exception}"
            "\n\n"
            "You can suppress this exception and fall back to eager by setting:\n"
            "    torchdynamo.config.raise_on_backend_error = False"
        )


class Unsupported(RuntimeError):
    def __init__(self, msg):
        super(Unsupported, self).__init__(msg)
        self.real_stack = []
        self.msg = msg
        self.category = None
        self.add_to_stats()

    def __str__(self):
        msgs = [super(Unsupported, self).__str__()]
        if self.real_stack:
            msgs.append("\nProcessing original code:\n")
            msgs.extend(
                reversed(traceback.StackSummary.from_list(self.real_stack).format())
            )
        return "".join(msgs)

    def remove_from_stats(self):
        counters[self.category][self.msg] -= 1
        if counters[self.category][self.msg] <= 0:
            del counters[self.category][self.msg]

    def add_to_stats(self, category="unimplemented"):
        self.category = category
        counters[category][self.msg] += 1


def unimplemented(msg: str):
    assert msg != os.environ.get("BREAK", False)
    raise Unsupported(msg)


def warning(msg: str):
    counters["warnings"][msg] += 1
    assert msg != os.environ.get("BREAK", False)

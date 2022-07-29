from . import allowed_functions
from . import convert_frame
from . import resume_execution
from .eval_frame import disable
from .eval_frame import export
from .eval_frame import optimize
from .eval_frame import optimize_assert
from .eval_frame import reset_code
from .eval_frame import run
from .eval_frame import skip
from .utils import guard_failures
from .utils import orig_code_map
import torchdynamo

__all__ = [
    "optimize",
    "optimize_assert",
    "export",
    "run",
    "disable",
    "reset",
    "list_backends",
    "skip",
]


def reset():
    """Clear all compile caches and restore initial state"""
    for weak_code in convert_frame.input_codes.seen + convert_frame.output_codes.seen:
        code = weak_code()
        if code:
            reset_code(code)
    convert_frame.input_codes.clear()
    convert_frame.output_codes.clear()
    orig_code_map.clear()
    guard_failures.clear()
    resume_execution.ContinueExecutionCache.cache.clear()


def list_backends():
    """
    Return valid strings that can be passed to:
        @torchdynamo.optimize(<backend>)
        def foo(...):
           ....
    """
    from .optimizations import BACKENDS

    return [*sorted([*BACKENDS.keys(), "inductor"])]


def allow_in_graph(fn):
    """
    Customize which functions TorchDynamo will include in the generated
    graph.  Similar to torch.fx.wrap().

        torchdynamo.allow_in_graph(my_custom_function)

        @torchdynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = my_custom_function(x)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will capture a single graph containing my_custom_function().
    """
    if isinstance(fn, (list, tuple)):
        return [allow_in_graph(x) for x in fn]
    assert callable(fn), "allow_in_graph expects a callable"
    allowed_functions._allowed_function_ids.add(id(fn))
    allowed_functions._disallowed_function_ids.remove(id(fn))


def disallow_in_graph(fn):
    """
    Customize which functions TorchDynamo will exclude in the generated
    graph and force a graph break on.

        torchdynamo.disallow_in_graph(torch.sub)

        @torchdynamo.optimize(...)
        def fn(a):
            x = torch.add(x, 1)
            x = torch.sub(x, 1)
            x = torch.add(x, 1)
            return x

        fn(...)

    Will break the graph on torch.sub, and give two graphs each with a
    single torch.add() op.
    """
    if isinstance(fn, (list, tuple)):
        return [disallow_in_graph(x) for x in fn]
    assert callable(fn), "disallow_in_graph expects a callable"
    allowed_functions._allowed_function_ids.remove(id(fn))
    allowed_functions._disallowed_function_ids.add(id(fn))


def setup_summary():
    import git
    import torch
    import os
    from os.path import exists
    def print_commit_hash(path, name):
        if exists(path):
            repo = git.Repo(path, search_parent_directories=True)
            sha = repo.head.object.hexsha
            print(f"{name} commit: {sha}")
        else:
            print(f"{name} Absent")
    
    print("## Commit hashes ##")
    print_commit_hash(".", "torchdynamo")
    print_commit_hash("../pytorch", "pytorch")
    print_commit_hash("../functorch", "functorch")
    print_commit_hash("../torchbenchmark", "torchbench")

    print()
    print("## TorchDynamo config flags ##")
    for key in dir(torchdynamo.config):
        val = getattr(torchdynamo.config, key)
        if not key.startswith("__") and isinstance(val, bool):
            print(f"torchdynamo.config.{key} = {val}")
    print(f"torch: {torch.__version__}")
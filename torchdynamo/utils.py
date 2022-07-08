import collections
import contextlib
import copy
import dataclasses
import functools
import gc
import inspect
import itertools
import logging.config
import math
import operator
import os
import re
import sys
import time
import types
import weakref
from functools import lru_cache
from typing import Any
from typing import Dict

import numpy as np
import tabulate
import torch
from torch import fx
from torch.nn.modules.lazy import LazyModuleMixin

import torchdynamo.config

from . import config

log = logging.getLogger(__name__)
counters = collections.defaultdict(collections.Counter)
troubleshooting_url = (
    "https://github.com/pytorch/torchdynamo/blob/main/TROUBLESHOOTING.md"
)


LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "torchdynamo_format": {"format": "%(levelname)s %(name)s: %(message)s"},
    },
    "handlers": {
        "torchdynamo_console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "torchdynamo_format",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "torchdynamo": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
        "torchinductor": {
            "level": "DEBUG",
            "handlers": ["torchdynamo_console"],
            "propagate": False,
        },
    },
}


def count_calls(g: fx.Graph):
    c = 0
    for n in g.nodes:
        if "call" in n.op:
            c += 1
    return c


def identity(x):
    return x


def nothing(*args, **kwargs):
    pass


class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = dict()
        self.refs = dict()

    def __getitem__(self, key):
        return self.values[id(key)]

    def get(self, key, default=None):
        return self.values.get(id(key), default)

    def __contains__(self, key):
        return id(key) in self.values

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self):
        self.refs.clear()
        self.values.clear()


def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


def is_numpy_int_type(value):
    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


def is_numpy_float_type(value):
    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


def istensor(obj):
    """Check of obj is a tensor"""
    return istype(
        obj, (torch.Tensor, torch.nn.Parameter, *config.traceable_tensor_subclasses)
    )


def is_lazy_module(mod):
    return isinstance(mod, LazyModuleMixin)


@functools.lru_cache(4096)
def print_once(*args):
    print(*args)


def make_cell(val=None):
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val

    def f():
        return x

    assert len(f.__closure__) == 1
    return f.__closure__[0]


def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
        return proxy_args, proxy_kwargs
    except NotImplementedError:
        from .exc import unimplemented
        from .variables.base import typestr

        raise unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
        )


@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]
    name: str

    def __call__(self, *args):
        CleanupManager.count -= 1
        del self.scope[self.name]

    @staticmethod
    def create(scope, name, val):
        assert name not in scope
        CleanupManager.count += 1
        scope[name] = val
        return CleanupHook(scope, name)


class CleanupManager(ExactWeakKeyDictionary):
    count = 0

    def _remove_id(self, idx):
        for hook in self.values[idx]:
            hook()
        super()._remove_id(idx)


CleanupManager.instance = CleanupManager()


def clone_tensor(x):
    """Clone the tensor and its gradient"""
    y = x.clone().requires_grad_(x.requires_grad)
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()
    return y


def clone_input(x):
    """copy while preserving strides"""
    with torch.no_grad():
        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        buffer = torch.empty(needed_size + 32, dtype=x.dtype, device=x.device)
        cache_line_offset = (
            (x.data_ptr() - buffer.data_ptr()) % 32
        ) // x.element_size()
        result = torch.as_strided(buffer, x.size(), x.stride(), cache_line_offset)
        try:
            result.copy_(x.clone())
        except RuntimeError:
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch.clone(x)
        return result


def clone_inputs(example_inputs):
    if isinstance(example_inputs, dict):
        res = dict(example_inputs)
        for key, value in res.items():
            assert isinstance(value, torch.Tensor)
            res[key] = clone_input(value)
        return res

    res = list(example_inputs)
    for i in range(len(res)):
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


def torchscript(model, example_inputs, verbose=True):
    if is_jit_model(model):
        # already done?
        return model

    try:
        return torch.jit.trace(model, example_inputs)
    except Exception:
        if verbose:
            log.exception("jit error")
        try:
            return torch.jit.script(model)
        except Exception:
            if verbose:
                log.exception("jit error")
    return None


def getfile(obj):
    try:
        return inspect.getfile(obj)
    except TypeError:
        return None


def is_namedtuple(obj):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


def is_namedtuple_cls(cls):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    try:
        if issubclass(cls, tuple):
            bases = getattr(cls, "__bases__", []) or [None]
            module = getattr(cls, "__module__", None)
            return module == "torch.return_types" or (
                bases[0] is tuple and hasattr(cls, "_make") and hasattr(cls, "_fields")
            )
    except TypeError:
        pass
    return False


@functools.lru_cache(1)
def namedtuple_fields(cls):
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    if cls is slice:
        return ["start", "stop", "step"]

    assert issubclass(cls, tuple)
    if hasattr(cls, "_fields"):
        # normal namedtuples
        return cls._fields

    @dataclasses.dataclass
    class Marker:
        index: int

    # frustrating ones e.g. torch.return_types.max
    assert cls.__module__ == "torch.return_types"
    obj = cls(map(Marker, range(cls.n_fields)))
    fields = [None] * cls.n_fields
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[getattr(obj, name).index] = name
    return fields


def checkpoint_params(gm):
    with torch.no_grad():
        rng_state = torch.clone(torch.random.get_rng_state())
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = []
        for param in itertools.chain(gm.parameters(), gm.buffers()):
            saved_state.append((param, param._version, torch.clone(param)))

    def restore():
        with torch.no_grad():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


def timed(model, example_inputs, times=1):
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        synchronize = nothing

    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0


def check_is_cuda(gm, example_inputs):
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))


@lru_cache(32)
def rot_n_helper(n):
    assert n > 1
    vars = [f"v{i}" for i in range(n)]
    rotated = reversed(vars[-1:] + vars[:-1])
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"
    return fn


def is_safe_constant(v):
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    return istype(
        v, (types.CodeType, int, float, bool, str, bytes, type(None), slice, type(type))
    )


def check_constant_args(args, kwargs):
    return all(x.is_python_constant() for x in itertools.chain(args, kwargs.values()))


dict_values = type(dict().values())
odict_values = type(collections.OrderedDict().values())
tuple_iterator = type(iter(tuple()))
tuple_iterator_len = tuple_iterator.__length_hint__
object_new = object.__new__


def product(it):
    return functools.reduce(operator.mul, it, 1)


def tuple_iterator_getitem(it, index):
    _, (obj,), start = it.__reduce__()
    return obj[start + index]


def rename_implicit(v):
    """
    Usage of inline comprehensions generates a implicit ".0" variable that
    trips up guard generation.  This renames these variables in guards.
    """
    m = re.match(r"^[.](\d+)$", v)
    if m:
        assert v == ".0", f"currently only .0 supported: {v}"
        # to support .1 etc see guards.py and _eval_frame.c
        return f"___implicit{m.group(1)}"
    return v


# FakeTensors were introduced after pytorch 1.12, so gate their use
# to allow pytorch 1.12 to work
fake_tensors_available = True
try:
    from torch._subclasses import FakeTensorMode  # noqa: F401
    from torch._subclasses import UnsupportedFakeTensorException

    def wrap_fake_exception(fn):
        try:
            return fn()
        except UnsupportedFakeTensorException as e:
            raise torchdynamo.exc.FakeTensorError(
                f"Unsupported: {e.reason} with fake tensor propagation. "
                "Run with config.fake_tensor_propagation=False"
            ) from e

    def wrap_to_fake_tensor(e, fake_mode):
        if type(e) in (torch.Tensor, torch.nn.Parameter):
            return wrap_fake_exception(lambda: fake_mode.from_tensor(e))
        else:
            return e

    def deepcopy_to_fake_tensor(obj, fake_mode):
        with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
            return wrap_fake_exception(lambda: copy.deepcopy(obj))

except ImportError:
    fake_tensors_available = False


def same(a, b, cos_similarity=False, tol=1e-4, equal_nan=False):
    """Check correctness to see if a and b match"""
    if isinstance(a, (list, tuple, torch.nn.ParameterList, torch.Size)):
        assert isinstance(b, (list, tuple)), f"type mismatch {type(a)} {type(b)}"
        return len(a) == len(b) and all(
            same(ai, bi, cos_similarity, tol, equal_nan) for ai, bi in zip(a, b)
        )
    elif isinstance(a, dict):
        assert isinstance(b, dict)
        assert set(a.keys()) == set(
            b.keys()
        ), f"keys mismatch {set(a.keys())} == {set(b.keys())}"
        for k in a.keys():
            if not (same(a[k], b[k], cos_similarity, tol, equal_nan=equal_nan)):
                print("Accuracy failed for key name", k)
                return False
        return True
    elif isinstance(a, torch.Tensor):
        if a.is_sparse:
            assert b.is_sparse
            a = a.to_dense()
            b = b.to_dense()
        assert isinstance(b, torch.Tensor), f"type mismatch {type(a)} {type(b)}"
        if cos_similarity:
            # TRT will bring error loss larger than current threshold. Use cosine similarity as replacement
            a = a.flatten().to(torch.float32)
            b = b.flatten().to(torch.float32)
            res = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-6)
            if res < 0.99:
                print(f"Similarity score={res.cpu().detach().item()}")
            return res >= 0.99
        else:
            return torch.allclose(a, b, atol=tol, rtol=tol, equal_nan=equal_nan)
    elif isinstance(a, (str, int, type(None), bool, torch.device)):
        return a == b
    elif isinstance(a, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    elif is_numpy_int_type(a) or is_numpy_float_type(a):
        return (type(a) is type(b)) and (a == b)
    elif type(a).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        assert type(a) is type(b)
        return all(
            same(getattr(a, key), getattr(b, key), cos_similarity, tol, equal_nan)
            for key in a.__dict__.keys()
        )
    else:
        raise RuntimeError(f"unsupported type: {type(a).__name__}")


def format_func_info(code):
    short_filename = code.co_filename.split("/")[-1]
    return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"


@contextlib.contextmanager
def disable_cache_limit():
    prior = torchdynamo.config.cache_size_limit
    torchdynamo.config.cache_size_limit = sys.maxsize

    try:
        yield
    finally:
        pass
        torchdynamo.config.cache_size_limit = prior


# map from transformed code back to original user code
orig_code_map = ExactWeakKeyDictionary()

# keep a record of code_obj -> list of guard failure reasons for logging
guard_failures = collections.defaultdict(list)


class CompileProfiler:
    """Utility for profiling how and what dynamo would compile.

    Can be used for
     * diagnosing recompilation issues
     * determining an appropriate compile cache limit
     * (TODO)confirming which functions got compiled/skipped
    """

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.backend_ctx_ctor = lambda: disable_cache_limit()

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def get_metrics(self):
        return {"guard_failures": guard_failures}

    def report(self):
        metrics = self.get_metrics()
        gf = metrics["guard_failures"]

        def num_recompiles(code):
            return len(gf[code])

        def recompile_reasons(code):
            return "\n".join([str(x) for x in gf[code]])

        summarized_gf = [
            [format_func_info(code), num_recompiles(code), recompile_reasons(code)]
            for code in gf
        ]
        rpt = "Torchdynamo Profiler Report\n"
        if "graph_break" in counters:
            rpt += "\n"
            rpt += "The following conditions caused torchdynamo to break out of tracing and fall back to python.\n"
            rpt += (
                "You may gain additional insight by passing `nopython=True` to torchdynamo.optimize, "
                "to break on the first condition.\n"
            )
            graph_breaks = counters["graph_break"]
            rpt += tabulate.tabulate(
                [[msg, graph_breaks[msg]] for msg in graph_breaks],
                headers=["Graph Break Reason", "Count"],
            )

        if len(gf):
            max_recompiles = max([num_recompiles(code) for code in gf])
            rpt += "\n"
            rpt += (
                "These subgraphs were recompiled more than once due to guard failures."
            )
            rpt += (
                "Guard failures indicate some condition assumed to be static by the tracer changed, "
                "making it unsafe to reuse the compiled program."
            )
            rpt += tabulate.tabulate(
                summarized_gf,
                headers=["Function", "Num Recompiles", "Recompile Reasons"],
            )
            rpt += "\n"
            rpt += f"Set torchdynamo.config.cache_size_limit to {max_recompiles} to avoid being cache limited.\n"
        else:
            rpt += "No cache-limited recompilations detected.\n"

        return rpt

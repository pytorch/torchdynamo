import functools
import operator

import sympy
import torch


@functools.lru_cache(None)
def has_triton():
    try:
        import triton

        return triton is not None
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_torchvision_roi_align():
    try:
        from torchvision.ops import roi_align  # noqa

        return roi_align is not None and hasattr(
            getattr(torch.ops, "torchvision", None), "roi_align"
        )
    except (ImportError, ModuleNotFoundError):
        return False


@functools.lru_cache(None)
def has_triton_libdevice():
    try:
        from triton.language import libdevice

        return libdevice is not None
    except (ImportError, ModuleNotFoundError):
        return False


def conditional_product(*args):
    return functools.reduce(operator.mul, [x for x in args if x])


def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))


def sympy_dot(seq1, seq2):
    assert len(seq1) == len(seq2)
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def unique(it):
    return {id(x): x for x in it}.values()


def gen_gm_and_inputs(target, args, kwargs):
    g = torch.fx.Graph()
    g_args = []
    a_args = []
    for n, arg in enumerate(args):
        if isinstance(arg, torch.Tensor):
            g_args.append(g.placeholder(f"arg{n}"))
            a_args.append(arg)
        else:
            g_args.append(arg)
    assert all(not isinstance(x, torch.Tensor) for x in kwargs.values())
    node = g.call_function(target, tuple(g_args), kwargs)
    if (
        len(target._schema.returns) == 1
        and str(target._schema.returns[0].type) == "Tensor"
    ):
        node = (node,)
    g.output(node)

    gm = torch.fx.GraphModule({}, g)
    return gm, a_args

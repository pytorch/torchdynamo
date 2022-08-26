import dataclasses
import functools
import logging
import operator
from typing import List

import torch.fx
from functorch.compile import min_cut_rematerialization_partition

from torchdynamo.debug_utils import wrap_debug
from torchdynamo.optimizations.backends import aot_autograd
from torchdynamo.optimizations.normalize import normalize_ir
from torchdynamo.testing import same
from torchdynamo.utils import identity
from torchdynamo.utils import init_logging

from . import config
from . import overrides
from .decomposition import select_decomp_table
from .graph import GraphLowering
from .utils import ceildiv
from .utils import gen_gm_and_inputs
from .virtualized import V

log = logging.getLogger(__name__)


@dataclasses.dataclass
class BoxedBool:
    value: bool

    def __bool__(self):
        return self.value

    @staticmethod
    def disable(obj):
        if isinstance(obj, BoxedBool):
            obj.value = False
            return obj
        return False


class CheckEachNode(torch.fx.Interpreter):
    def call_function(self, target, args, kwargs):
        expected = target(*args, **kwargs)
        if target in (operator.getitem,):
            return expected

        gm, gm_inps = gen_gm_and_inputs(target, args, kwargs)
        graph = GraphLowering(gm)
        with V.set_graph_handler(graph):
            graph.run(*args, **kwargs)
            actual = graph.compile_to_fn()(*gm_inps)

        if isinstance(expected, torch.Tensor):
            actual = actual[0]

        print(target, same(expected, actual))
        assert same(expected, actual)

        return expected


@functools.partial(wrap_debug, compiler_name="inductor")
def compile_fx_inner(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    wrap=identity,
    cudagraphs=None,
    num_fixed=0,
):
    init_logging()

    if cudagraphs is None:
        cudagraphs = config.triton.cudagraphs

    graph = GraphLowering(gm, num_dynamic_inputs=len(example_inputs))
    with V.set_graph_handler(graph):
        wrap(graph.run)(*example_inputs)
        compiled_fn = wrap(graph.compile_to_fn())

    if cudagraphs and set(graph.device_types) == {"cuda"} and not graph.mutated_inputs:
        compiled_fn = cudagraphify(
            compiled_fn, example_inputs, static_input_idxs=range(num_fixed)
        )
    elif cudagraphs:
        BoxedBool.disable(cudagraphs)

        if len(set(graph.device_types)) > 1:
            log.warning("skipping cudagraphs due to multiple devices")
        elif graph.mutated_inputs and set(graph.device_types) == {"cuda"}:
            log.warning("skipping cudagraphs due to input mutation")

    return compiled_fn


def cudagraphify(model, inputs, static_input_idxs=()):
    """
    Assumes inputs[static_input_idxs[i]] are always the same memory address
    """

    def static_input(x):
        # make sure alignment and contiguity of inputs is preserved
        needed_size = (
            sum((shape - 1) * stride for shape, stride in zip(x.size(), x.stride())) + 1
        )
        needed_size = ceildiv(needed_size, 32) * 32
        buffer = torch.zeros(needed_size, dtype=x.dtype, device=x.device)
        cache_line_offset = (
            (x.data_ptr() - buffer.data_ptr()) % 32
        ) // x.element_size()
        return torch.as_strided(buffer, x.size(), x.stride(), cache_line_offset)

    assert isinstance(inputs, (list, tuple))
    static_inputs = [
        static_input(x) if idx not in static_input_idxs else inputs[idx]
        for idx, x in enumerate(inputs)
    ]

    # warmup
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        model(*inputs)
    stream.synchronize()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()

    # record
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        static_outputs = model(*static_inputs)
    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    if config.size_asserts:

        def run(*new_inputs):
            assert len(static_inputs) == len(new_inputs)
            for idx, (dst, src) in enumerate(zip(static_inputs, new_inputs)):
                if idx in static_input_idxs:
                    assert dst.data_ptr() == src.data_ptr()
                else:
                    dst.copy_(src)
            graph.replay()
            return static_outputs

    else:
        copy_indices = [
            idx for idx in range(len(static_inputs)) if idx not in static_input_idxs
        ]

        def run(*new_inputs):
            for idx in copy_indices:
                static_inputs[idx].copy_(new_inputs[idx])
            graph.replay()
            return static_outputs

    return run


def count_tangents(fx_g: torch.fx.GraphModule):
    """
    Infers which inputs are static for a backwards graph
    """

    def is_not_gradout(x):
        return "tangents" not in x.name

    arg_count = 0
    static_arg_idxs = []
    for n in fx_g.graph.nodes:
        if n.op == "placeholder":
            if is_not_gradout(n):
                static_arg_idxs.append(arg_count)
            arg_count += 1

    assert static_arg_idxs == list(range(len(static_arg_idxs)))
    return len(static_arg_idxs)


def compile_fx(model_: torch.fx.GraphModule, example_inputs_: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""
    logging.getLogger("torchinductor").setLevel(
        logging.DEBUG if config.debug else logging.WARNING
    )

    with overrides.patch_functions():
        model_ = normalize_ir(model_, example_inputs_)
        model_ = overrides.replace_fx(model_)
    num_example_inputs = len(example_inputs_)
    cudagraphs = BoxedBool(config.triton.cudagraphs)

    def fw_compiler(model: torch.fx.GraphModule, example_inputs):
        if config.debug:
            print("FORWARD GRAPH:")
            model.graph.print_tabular()
        fixed = len(example_inputs) - num_example_inputs
        return compile_fx_inner(
            model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs
        )

    def bw_compiler(model: torch.fx.GraphModule, example_inputs):
        if config.debug:
            print("BACKWARD GRAPH:")
            model.graph.print_tabular()
        fixed = count_tangents(model)
        return compile_fx_inner(
            model, example_inputs, num_fixed=fixed, cudagraphs=cudagraphs
        )

    with overrides.patch_functions():
        return aot_autograd(
            model_,
            example_inputs_,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            decompositions=select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )

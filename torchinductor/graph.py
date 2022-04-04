import collections
from itertools import chain

import torch
import torch.fx
from sympy import Integer

from . import config
from .codegen.schedule import ScheduleCodeGen
from .ir import Constant
from .ir import FixedLayout
from .ir import InputBuffer
from .ir import TensorBox
from .lowering import lowerings
from .sizevars import SizeVarAllocator


class GraphLowering(torch.fx.Interpreter):
    def symbolic_sizes_strides(self, ex: torch.Tensor):
        """
        Support dynamic shapes and dynamic strides by assigning variables
        to each dimension.  We duck-shape tensors, so if two tensors
        have the same size they get assigned the same symbolic variable.
        """
        size = [self.sizevars[i] for i in ex.size()]
        stride = [None] * len(size)
        for i, val in enumerate(ex.stride()):
            if val in (0, 1):
                stride[i] = Integer(val)

        while any(x is None for x in stride):
            candidates = {
                ex.size(i) * ex.stride(i): size[i] * stride[i]
                for i in range(len(size))
                if stride[i] is not None
            }
            for i in chain(reversed(range(len(stride))), range(len(stride))):
                if stride[i] is None and ex.stride(i) in candidates:
                    stride[i] = candidates[ex.stride(i)]
                    candidates[ex.size(i) * ex.stride(i)] = size[i] * stride[i]
            if any(x is None for x in stride):
                # bind the smallest unbound stride to a new variable
                val, i = sorted(
                    [(ex.stride(i), i) for i in range(len(stride)) if stride[i] is None]
                )[0]
                stride[i] = self.sizevars[val]

        # print(f"{ex.size()} = {size}, {ex.stride()} = {stride}")
        return size, stride

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.sizevars = SizeVarAllocator("s")
        self.graph_inputs = collections.OrderedDict()
        self.graph_outputs = None
        self.device = None

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        if self.device is None:
            self.device = example.device
        assert example.device == self.device
        sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        data = InputBuffer(
            FixedLayout(example.device, example.dtype, sizes, strides),
            len(self.graph_inputs),
            target,
        )
        tensor = TensorBox.create(data)
        self.graph_inputs[target] = tensor
        return tensor

    def call_function(self, target, args, kwargs):
        return lowerings[target](*args, **kwargs)

    def get_attr(self, target, args, kwargs):
        # this is a constant
        value = getattr(self.module, target)
        assert value.shape == ()
        return Constant(value.item(), value.dtype, value.device)

    def call_module(self, target, args, kwargs):
        assert False

    def output(self, target, args, kwargs):
        result = super().output(target, args, kwargs)
        assert isinstance(result, tuple)
        self.graph_outputs = collections.OrderedDict(
            (f"out{i}", r) for i, r in enumerate(result)
        )

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        num_users = len(set(n.users))
        if num_users > 1:
            # TODO(jansel): introduce a store vs inline choice
            result.mark_reuse(n.users)
        return result

    def codegen(self):
        from torchinductor.codegen.cpp import CppPointwiseKernel
        from torchinductor.codegen.triton import TritonPointwiseKernel

        schedule = ScheduleCodeGen(self)

        backends = {"cpu": CppPointwiseKernel, "cuda": TritonPointwiseKernel}
        backend_cls = backends[self.device.type]
        backend_cls.codegen(self, self.graph_outputs, schedule)
        # TODO(jansel): manage a dependency graph
        return schedule.generate()

    def compile_to_fn(self):
        from .codecache import PyCodeCache

        code = self.codegen()
        if config.debug:
            print(code)
        return PyCodeCache.load(code).call

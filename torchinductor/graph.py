import collections
import operator
from itertools import chain

import sympy
import torch
import torch.fx
from sympy import Integer

from . import config
from . import ir
from .codegen.wrapper import WrapperCodeGen
from .exc import LoweringException
from .exc import MissingOperator
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
                if stride[i] is not None and ex.stride(i) >= 0
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
        return size, stride

    def static_sizes_strides(self, ex: torch.Tensor):
        """
        Primarily used to weights
        """
        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def __init__(self, gm: torch.fx.GraphModule, num_dynamic_inputs):
        super().__init__(gm)
        self.sizevars = SizeVarAllocator("s")
        self.graph_inputs = collections.OrderedDict()
        self.graph_outputs = None
        self.device = None
        self.buffers = []
        self.removed_buffers = set()
        self.wrapper_code = None
        self.num_dynamic_inputs = num_dynamic_inputs
        self.num_static_inputs = None

    def run(self, *args):
        self.num_static_inputs = len(args) - self.num_dynamic_inputs
        return super().run(*args)

    def register_buffer(self, buffer: ir.ComputedBuffer):
        name = f"buf{len(self.buffers)}"
        self.buffers.append(buffer)
        return name

    def placeholder(self, target, args, kwargs):
        example: torch.Tensor = super().placeholder(target, args, kwargs)
        if self.device is None:
            self.device = example.device
        assert example.device == self.device
        if config.static_weight_shapes and (
            len(self.graph_inputs) < self.num_static_inputs or not config.dynamic_shapes
        ):
            # the first N inputs are weights
            sizes, strides = self.static_sizes_strides(example)
        else:
            sizes, strides = self.symbolic_sizes_strides(example)
        # TODO(jansel): handle input aliasing
        tensor = TensorBox.create(
            InputBuffer(
                target,
                FixedLayout(example.device, example.dtype, sizes, strides),
            )
        )
        self.graph_inputs[target] = tensor
        return tensor

    def call_function(self, target, args, kwargs):
        if target is operator.getitem and isinstance(args[0], (list, tuple)):
            return super().call_function(target, args, kwargs)

        if target not in lowerings:
            raise MissingOperator(target, args, kwargs)
        try:
            return lowerings[target](*args, **kwargs)
        except Exception as e:
            raise LoweringException(e, target, args, kwargs) from e

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
        assert all(isinstance(x, TensorBox) for x in result), result
        self.graph_outputs = [ir.ExternKernel.realize_input(x) for x in result]

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)
        num_users = len(set(n.users))
        if num_users > 1 and isinstance(result, TensorBox):
            # TODO(jansel): introduce a store vs inline choice
            result.mark_reuse(len(n.users))
        return result

    def codegen(self):
        from .codegen.cpp import CppKernel
        from .codegen.triton import TritonKernel

        wrapper = WrapperCodeGen()
        self.wrapper_code = wrapper

        backends = {"cpu": CppKernel, "cuda": TritonKernel}
        backend_cls = backends[self.device.type]
        backend_cls.codegen(wrapper)
        # TODO(jansel): manage a dependency graph
        return wrapper.generate()

    def compile_to_fn(self):
        from .codecache import PyCodeCache

        code = self.codegen()
        if config.debug:
            print(code)
        return PyCodeCache.load(code).call

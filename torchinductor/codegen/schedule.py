from itertools import chain
from itertools import count

from torchinductor import codecache
from torchinductor.codegen.common import CodeGen
from torchinductor.codegen.common import IndentedBuffer
from torchinductor.codegen.common import Kernel


class ScheduleCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels above.
    """

    def __init__(self, graph):
        super().__init__()
        self._names_iter = count()
        self.graph = graph
        self.header = IndentedBuffer()
        self.body = IndentedBuffer(initial_indent=1)
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                from torch import empty, empty_like
                # TODO(jansel): handle triton missing
                import triton
                from triton import cdiv
                import triton.language as tl
                from {codecache.__name__} import CppCodeCache, TritonCodeCache, grid
            """
        )
        with self.body.indent(-1):
            self.body.writelines(
                [f"def call({', '.join(self.graph.graph_inputs.keys())}):"]
            )
        self.graph.sizevars.codegen(self.body, self.graph.graph_inputs)
        self.codegen_outputs()

    def next_kernel_name(self):
        return f"kernel{next(self._names_iter)}"

    def codegen_outputs(self):
        code = self.body
        empty_like_cache = dict()
        for name, value in self.graph.graph_inputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            empty_like_cache.setdefault((device, dtype, shape, stride), name)

        for name, value in self.graph.graph_outputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            key = (device, dtype, shape, stride)
            # TODO(jansel): strides?
            if key in empty_like_cache:
                code.writeline(f"{name} = empty_like({empty_like_cache[key]})")
            else:
                code.writeline(
                    f"{name} = empty([{', '.join(map(str, shape))}], device='{device.type}', dtype={dtype})"
                )
        for name, value in chain(
            self.graph.graph_inputs.items(), self.graph.graph_outputs.items()
        ):
            device = value.get_device()
            if device.type == "cpu":
                code.writeline(f"{name}_ptr = c_void_p({name}.data_ptr())")

    def generate(self):
        self.body.writeline(
            "return (" + ", ".join(self.graph.graph_outputs.keys()) + ", )"
        )
        return f"{self.header.getvalue()}\n\n{self.body.getvalue()}"

    def define_kernel(self, name: str, kernel: str):
        self.header.splice(f"\n\n{name} = {kernel}")

    def call_kernel(self, name: str, kernel: Kernel):
        kernel.call_kernel(self, self.body, name)

from itertools import count

from .. import codecache
from ..virtualized import graph
from .common import CodeGen
from .common import IndentedBuffer
from .common import Kernel


class WrapperCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels above.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.body = IndentedBuffer()
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                from torch import empty, empty_like
                from {codecache.__name__} import CppCodeCache, TritonCodeCache, grid
                try:
                    import triton
                    from triton import cdiv
                    import triton.language as tl
                except ImportError:
                    pass
            """
        )
        self.prefix.writelines(
            ["", "", f"def call({', '.join(graph.graph_inputs.keys())}):"]
        )
        with self.prefix.indent():
            graph.sizevars.codegen(self.prefix, graph.graph_inputs)

    def next_kernel_name(self):
        return f"kernel{next(self._names_iter)}"

    def codegen_outputs(self, code):
        for value in graph.graph_inputs.values():
            name = value.get_name()
            device = value.get_device()
            if device.type == "cpu":
                code.writeline(f"{name}_ptr = c_void_p({name}.data_ptr())")

        empty_like_cache = dict()
        for name, value in graph.graph_inputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            empty_like_cache.setdefault((device, dtype, shape, stride), name)

        for buffer in graph.buffers:
            name = buffer.get_name()
            if name in graph.removed_buffers:
                continue
            device = buffer.get_device()
            dtype = buffer.get_dtype()
            shape = tuple(buffer.get_size())
            stride = tuple(buffer.get_stride())
            key = (device, dtype, shape, stride)
            if key in empty_like_cache:
                code.writeline(f"{name} = empty_like({empty_like_cache[key]})")
            else:
                code.writeline(
                    f"{name} = empty([{', '.join(map(str, shape))}], device='{device.type}', dtype={dtype})"
                )

            if device.type == "cpu":
                code.writeline(f"{name}_ptr = c_void_p({name}.data_ptr())")

    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)
        result.splice(self.prefix)
        with result.indent():
            self.codegen_outputs(result)
            result.splice(self.body)
            result.writeline("return (" + ", ".join(graph.graph_outputs) + ", )")
        return result.getvalue()

    def define_kernel(self, name: str, kernel: str):
        self.header.splice(f"\n\n{name} = {kernel}")

    def call_kernel(self, name: str, kernel: Kernel):
        kernel.call_kernel(self, self.body, name)

from itertools import count

from .. import codecache
from .. import ir
from ..virtualized import V
from .common import CodeGen
from .common import IndentedBuffer
from .common import Kernel
from .triton import texpr

pexpr = texpr


class WrapperCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels.
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
                from torch import empty, empty_like, as_strided
                from {codecache.__name__} import CppCodeCache, TritonCodeCache, grid
                try:
                    import triton
                    from triton import cdiv
                    import triton.language as tl
                except ImportError:
                    pass
                aten = torch.ops.aten
            """
        )
        self.prefix.writelines(
            ["", "", f"def call({', '.join(V.graph.graph_inputs.keys())}):"]
        )
        with self.prefix.indent():
            V.graph.sizevars.codegen(self.prefix, V.graph.graph_inputs)

        empty_like_cache = dict()
        for name, value in V.graph.graph_inputs.items():
            device = value.get_device()
            dtype = value.get_dtype()
            shape = tuple(value.get_size())
            stride = tuple(value.get_stride())
            empty_like_cache.setdefault((device, dtype, shape, stride), name)
        self.empty_like_cache = empty_like_cache
        self.allocated = set()

    def next_kernel_name(self):
        return f"kernel{next(self._names_iter)}"

    def codegen_allocation(self, buffer):
        name = buffer.get_name()
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(layout.view, ir.ReinterpretView)
            self.codegen_allocation(layout.view.data)
            self.body.writeline(f"{name} = {layout.view.codegen_reference()}")
            return

        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        key = (device, dtype, shape, stride)
        if key in self.empty_like_cache:
            self.body.writeline(f"{name} = empty_like({self.empty_like_cache[key]})")
        else:
            self.body.writeline(
                f"{name} = empty([{', '.join(map(pexpr, shape))}], device='{device.type}', dtype={dtype})"
            )

    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)
        result.splice(self.prefix)
        with result.indent():
            result.splice(self.body)
            output_refs = [x.codegen_reference() for x in V.graph.graph_outputs]
            result.writeline("return (" + ", ".join(output_refs) + ", )")
        return result.getvalue()

    def define_kernel(self, name: str, kernel: str):
        self.header.splice(f"\n\n{name} = {kernel}")

    def call_kernel(self, name: str, kernel: Kernel):
        kernel.call_kernel(self, self.body, name)

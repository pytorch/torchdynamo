import collections
import dataclasses
import hashlib
from itertools import count

from .. import codecache
from .. import config
from .. import ir
from ..utils import has_triton
from ..utils import sympy_product
from ..virtualized import V
from .common import CodeGen
from .common import DeferredLine
from .common import IndentedBuffer
from .common import Kernel
from .triton import texpr

pexpr = texpr


def buffer_reuse_key(node: ir.Buffer):
    return (node.get_device(), node.get_dtype(), sympy_product(node.get_size()))


def make_buffer_reuse(old, new):
    assert old.get_dtype() == new.get_dtype()
    if old.get_size() == new.get_size() and old.get_stride() == new.get_stride():
        return f"{new.get_name()} = {old.get_name()}; del {old.get_name()}"

    return (
        f"{new.get_name()} = as_strided({old.get_name()}, "
        f"{V.graph.sizevars.codegen_shape_tuple(new.get_size())}, "
        f"{V.graph.sizevars.codegen_shape_tuple(new.get_stride())}); del {old.get_name()}"
    )


@dataclasses.dataclass
class FreedBuffer:
    node: ir.Buffer
    reused_as: ir.Buffer = None

    def codegen(self, code: IndentedBuffer):
        name = self.node.get_name()
        if self.reused_as is not None:
            code.writeline(make_buffer_reuse(self.node, self.reused_as))
        else:
            code.writeline(f"del {name}")


class WrapperCodeGen(CodeGen):
    """
    The outer wrapper that calls the kernels.
    """

    def __init__(self):
        super().__init__()
        self._names_iter = count()
        self.header = IndentedBuffer()
        self.prefix = IndentedBuffer()
        self.lines = []
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
                import torch
                import random
                from torch import empty_strided, as_strided
                from {codecache.__name__} import CppCodeCache, TritonCodeCache

                aten = torch.ops.aten

            """
        )

        if has_triton():
            self.header.splice(
                """
                    import triton
                    import triton.language as tl

                    from torchinductor.triton_ops.autotune import pointwise_heuristics
                    from torchinductor.triton_ops.autotune import reduction_heuristics
                    from torchinductor.triton_ops.autotune import grid

                """
            )

            if config.triton.use_mm:
                self.header.writeline(
                    "from torchinductor.triton_ops.matmul import matmul_out as triton_mm_out"
                )

            if config.triton.use_bmm:
                self.header.writeline(
                    "from torchinductor.triton_ops.batched_matmul import bmm_out as triton_bmm_out"
                )

        self.prefix.writelines(
            ["", "", f"def call({', '.join(V.graph.graph_inputs.keys())}):"]
        )
        with self.prefix.indent():
            for name in V.graph.randomness_seeds:
                self.prefix.writeline(
                    f"torch.randint(2**31, size=(), dtype=torch.int32, out={name})"
                )
            V.graph.sizevars.codegen(self.prefix, V.graph.graph_inputs)

        for name, value in V.graph.constants.items():
            # include a hash so our code cache gives different constants different files
            hashed = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()
            self.header.writeline(f"{name} = None  # {hashed}")

        self.allocated = set()
        self.freed = set()
        self.reuse_pool = collections.defaultdict(list)

    def next_kernel_name(self):
        return f"kernel{next(self._names_iter)}"

    def codegen_allocation(self, buffer):
        name = buffer.get_name()
        if name in V.graph.removed_buffers or name in self.allocated:
            return
        self.allocated.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, ir.MutationLayout):
            return
        if isinstance(layout, ir.AliasedLayout):
            assert isinstance(layout.view, ir.ReinterpretView)
            self.codegen_allocation(layout.view.data)
            allocation = DeferredLine(
                name, f"{name} = {layout.view.codegen_reference()}"
            )
            self.writeline(allocation)
            return

        # try to reuse a recently freed buffer
        key = buffer_reuse_key(buffer)
        if self.reuse_pool.get(key, None):
            fb = self.reuse_pool[key].pop()
            fb.reused_as = buffer
            return

        device = buffer.get_device()
        dtype = buffer.get_dtype()
        shape = tuple(buffer.get_size())
        stride = tuple(buffer.get_stride())
        allocation = DeferredLine(
            name,
            f"{name} = empty_strided("
            f"{V.graph.sizevars.codegen_shape_tuple(shape)}, "
            f"{V.graph.sizevars.codegen_shape_tuple(stride)}, "
            f"device='{device.type}', dtype={dtype})",
        )
        self.writeline(allocation)

    def codegen_free(self, buffer):
        name = buffer.get_name()
        if not self.can_reuse(buffer):
            return
        self.freed.add(name)

        layout = buffer.get_layout()
        if isinstance(layout, (ir.AliasedLayout, ir.MultiOutputLayout)):
            self.writeline(f"del {name}")
            return

        fb = FreedBuffer(buffer)
        self.reuse_pool[buffer_reuse_key(buffer)].append(fb)
        self.writeline(fb)

    def can_reuse(self, buffer):
        name = buffer.get_name()
        if (
            name in V.graph.removed_buffers
            or name in V.graph.graph_inputs
            or name in V.graph.constants
            or name in self.freed
        ):
            return False
        return True

    def codegen_inplace_reuse(self, input_buffer, output_buffer):
        assert buffer_reuse_key(input_buffer) == buffer_reuse_key(output_buffer)
        self.codegen_allocation(input_buffer)
        self.freed.add(input_buffer.get_name())
        self.allocated.add(output_buffer.get_name())
        self.writeline(FreedBuffer(input_buffer, output_buffer))

    def generate(self):
        result = IndentedBuffer()
        result.splice(self.header)
        result.splice(self.prefix)
        with result.indent():
            while self.lines and isinstance(self.lines[-1], FreedBuffer):
                self.lines.pop()
            for line in self.lines:
                if isinstance(line, FreedBuffer):
                    line.codegen(result)
                else:
                    result.writeline(line)
            output_refs = [x.codegen_reference() for x in V.graph.graph_outputs]
            if output_refs:
                result.writeline("return (" + ", ".join(output_refs) + ", )")
            else:
                result.writeline("return ()")

        self.add_benchmark_harness(result)

        return result.getvalue()

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        if not config.benchmark_harness:
            return
        output.writelines(["", "", 'if __name__ == "__main__":'])
        with output.indent():
            output.splice(
                """
                from torchdynamo.testing import rand_strided
                from benchmarks.microbenchmarks.microbench import print_performance
                """,
                strip=True,
            )
            for name, value in V.graph.graph_inputs.items():
                shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                device = value.get_device()
                dtype = value.get_dtype()
                output.writeline(
                    f"{name} = rand_strided("
                    f"{V.graph.sizevars.codegen_shape_tuple(shape)}, "
                    f"{V.graph.sizevars.codegen_shape_tuple(stride)}, "
                    f"device='{device.type}', dtype={dtype})"
                )
            output.writeline(
                f"print_performance(lambda: call({', '.join(V.graph.graph_inputs.keys())}))"
            )

    def define_kernel(self, name: str, kernel: str):
        self.header.splice(f"\n\n{name} = {kernel}")

    def call_kernel(self, name: str, kernel: Kernel):
        tmp = IndentedBuffer()
        kernel.call_kernel(self, tmp, name)
        for line in tmp.getvalue().split("\n"):
            line = line.strip()
            if line:
                self.writeline(line)

    def writeline(self, line):
        self.lines.append(line)

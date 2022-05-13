import contextlib

import sympy

from .. import ir
from ..scheduler import Scheduler
from ..virtualized import V
from ..virtualized import ops
from .common import BracesBuffer
from .common import IndentedBuffer
from .common import Kernel
from .common import KernelArgs
from .cpp import CppOverrides
from .cpp import CppPrinter

cexpr = CppPrinter().doprint


class JiteratorOverrides(CppOverrides):
    """Map element-wise ops to CUDA code"""

    @staticmethod
    def minimum(a, b):
        return f"::min({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"::max({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        if val == float("inf"):
            return "POS_INFINITY"
        elif val == float("-inf"):
            return "NEG_INFINITY"
        return ops.to_dtype(repr(val), dtype)


class JiteratorKernel(Kernel):
    overrides = JiteratorOverrides
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args):
        super(JiteratorKernel, self).__init__(args)
        self.call_ranges = None
        self.ranges = None
        self.itervars = None
        self.reduction_depth = None
        self.reduction_vars = {}

    # Jiterator handles indexing internally
    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        return self.cse.generate(self.loads, f"{var}")

    # Jiterator handles indexing internally
    def store(self, name: str, index, value):
        assert "buf" in name
        self.args.output(name)  # create a output name
        self.stores.writeline(f"return {value};")

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [sympy.Symbol(f"i{n}") for n in range(len(self.ranges))]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def codegen_kernel(self, code):
        with contextlib.ExitStack():
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)

    @classmethod
    def codegen(cls, wrapper):
        def codegen_extern_call(node: ir.ExternKernel):
            nonlocal kernel_group
            kernel_group.codegen_define_and_call(wrapper)
            node.codegen(wrapper)
            scheduler.barrier()
            kernel_group = KernelGroup()

        scheduler = Scheduler(tuple, V.graph.buffers)
        kernel_group = KernelGroup()

        for group, reduction_group in scheduler.iter_runable_groups(
            codegen_extern_call
        ):
            if reduction_group:
                raise Exception("Jiteartor doesn't support reduction yet")
            with scheduler.kernel(kernel_group.new_kernel()) as kernel:
                vars, reduction_vars = kernel.set_ranges(group, reduction_group)

                # first any pointwise sharing same loops
                for node in scheduler.pop_group((group + reduction_group, ())):
                    node.run(vars, reduction_vars)
                    node.mark_fusable()

            kernel_group.finalize_kernel(kernel, scheduler)

        kernel_group.codegen_define_and_call(wrapper)


class JiteratorKernelArgs(KernelArgs):
    def argdefs(self):
        # TODO: reconsider type specialization
        argdefs = []
        for _, inner in self.input_buffers.items():
            # dtype = buffer_types[outer]
            # argdefs.append(f"const {DTYPE_TO_CPP[dtype]} {inner}")
            argdefs.append(f"const T {inner}")
        return argdefs


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = JiteratorKernelArgs()
        self.loops_code = BracesBuffer()
        self.stack = contextlib.ExitStack()
        self.count = 0

    def new_kernel(self):
        return JiteratorKernel(self.args)

    def finalize_kernel(self, new_kernel, scheduler):
        self.count += 1
        code = self.loops_code
        new_kernel.codegen_kernel(code)

        if not scheduler.runable_nodes and scheduler.pending_buffer_names:
            scheduler.barrier()

    def codegen_define_and_call(self, wrapper):
        self.stack.close()
        if self.count == 0:
            return

        argdefs = ",\n".ljust(25).join(self.args.argdefs())
        code = BracesBuffer()
        code.writelines([f"template <typename T> T kernel({argdefs})"])

        with code.indent():
            code.splice(self.loops_code)

        codecache_def = IndentedBuffer()
        codecache_def.writeline("'''")
        codecache_def.splice(code)
        codecache_def.writeline("'''")

        kernel_name = wrapper.next_kernel_name()
        wrapper.define_kernel(kernel_name, codecache_def.getvalue())

        # generate the code to call this
        args = self.args
        call_args = []
        for name in args.input_buffers.keys():
            call_args.append(f"{name}")

        output_args = []
        for name in args.output_buffers.keys():
            output_args.append(f"{name}")

        wrapper.header.writeline(
            f"jitted_{kernel_name} = torch.cuda.jiterator._create_jit_fn({kernel_name})",
        )

        wrapper.writeline(
            "{} = jitted_{}({})".format(
                output_args[0], kernel_name, ", ".join(call_args)
            )
        )

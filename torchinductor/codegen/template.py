from jinja2 import Environment, PackageLoader, select_autoescape

from .. import config
from ..virtualized import V
from .common import IndentedBuffer
from .common import product


class TritonTemplateKernel:
    def __init__(self, template_name="") -> None:
        env = Environment(
            loader = PackageLoader("torchinductor/codegen"),
            trim_blocks = True,
            lstrip_blocks = True,
        )
        self.template = env.get_template(template_name+".j2")

    def codegen_func(self, render_vars):

        code = IndentedBuffer()
        rendered = self.template.render(render_vars)
        code.splice(rendered)

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")

        return wrapper.getvalue()
    
    def call_func(self, code, name: str):
        # gen code to call func
        # e.g. func1(arg0, arg1, ...)
        _, call_args = self.args.python_argdefs()
        call_args = ", ".join(call_args)
        code.writeline(f"{name}({call_args})")

class TritonTemplateScheduling:
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def group_fn(self, sizes):
        return tuple(V.graph.sizevars.simplify(product(s)) for s in sizes)

    def codegen(self, *groups):
        wrapper = V.graph.wrapper_code
        scheduler = self.scheduler

        reschedule=[]
        with scheduler.kernel(TritonTemplateKernel(*groups)) as kernel:
            for _ in scheduler.iter_fixed_point():
                # scheduler.pop_group will keep iterating all reachable fusable nodes
                for node in scheduler.pop_group(groups):
                    # scheduler.maybe_remove_buffer(node, check_group=is_group_matching)
                    node.run(*kernel.set_ranges(*node.get_ranges()))
                    node.mark_fusable(broadcast_after_reduce=True)

                # consider fusable pointwise groups
                for node in scheduler.pop_group(groups):
                    sizes, _ = node.get_ranges()
                    split = split_sizes(sizes, groups[0])
                    if split is None:
                        reschedule.append(node)
                    else:
                        scheduler.maybe_remove_buffer(
                            node, check_group=is_group_matching
                        )
                        node.run(
                            *kernel.set_ranges(sizes[:split], sizes[split:], [])
                        )
                        node.mark_fusable()

        func_name = wrapper.next_func_name()
        wrapper.header.splice(kernel.codegen_func(render_vars))
        if config.triton.many_files:
            wrapper.define_kernel(func_name, kernel.codegen_func())
        else:
            wrapper.header.splice(kernel.codegen_func(func_name))
        kernel.call_func(wrapper, func_name)

        scheduler.enqueue(reschedule)
        scheduler.barrier()
        scheduler.maybe_free_buffers()

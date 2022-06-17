from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import config
from .. import ir
from ..virtualized import V
from .common import IndentedBuffer, Kernel
from .triton import TritonKernel


template_dict = {
    ir.Convolution: "triton_conv"
}
class TritonTemplateKernel(TritonKernel):
    def __init__(self, node: ir.ExternKernel, *groups):
        super(TritonTemplateKernel, self).__init__(*groups)
        template_name = template_dict[type(node)]
        env = Environment(
            loader = FileSystemLoader("torchinductor/codegen"),
            trim_blocks = True,
            lstrip_blocks = True,
        )
        self.template = env.get_template(template_name+".j2")

    def codegen_body(self):
        """
        get render_variables that to be put into the template
        to generate the final code
        """
        # TODO: codegen_body
        return 
    def codegen_kernel(self, name=None):

        code = IndentedBuffer()

        argdefs, _ = self.args.python_argdefs()

        if config.dynamic_shapes:
            maybe_const = ""
        else:
            maybe_const = ": tl.constexpr"

        for tree in self.range_trees:
            if isinstance(tree.kernel, TritonKernel) \
                and (tree.prefix != "r" or self.inside_reduction):
                # skip current TritonTemplateKernel, no extra argdefs is needed
                argdefs.append(f"{tree.prefix}numel{maybe_const}")

        for tree in self.range_trees:
            if isinstance(tree.kernel, TritonKernel) \
                and (tree.prefix != "r" or self.inside_reduction):
                argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        self.codegen_body()
        code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")

        return wrapper.getvalue()
    
    def call_kernel(self, code, name: str):
        # gen code to call kernel
        # e.g. kernel1(arg0, arg1, ...)
        _, call_args = self.args.python_argdefs()
        call_args = ", ".join(call_args)
        code.writeline(f"{name}({call_args})")

def template_codegen(scheduler, node):
    """
    codegen function for triton templates
    scheduler: Scheduler
    node: ExternKernelSchedulerNode
    """
    wrapper = V.graph.wrapper_code
    deivce, group = node.group

    reschedule = []
    with scheduler.kernel(TritonTemplateKernel(node.node, *group)) as kernel:
        # scheduler.pop_group will keep iterating all reachable fusable SchedulerNodes
        for node in scheduler.pop_group(group):
            # scheduler.maybe_remove_buffer(node, check_group=is_group_matching)
            node.run(*kernel.set_ranges(*node.get_ranges()))
            node.mark_fusable()

        # TODO: reduction

    kernel_name = wrapper.next_kernel_name()
    # code gen kernel
    wrapper.define_kernel(kernel_name, kernel.codegen_kernel())
    # code gen call to kernel
    kernel.call_kernel(wrapper, kernel_name)

    scheduler.enqueue(reschedule)  # TODO: consider reschedule
    scheduler.barrier()
    scheduler.maybe_free_buffers()
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .. import config
from .. import ir
from ..virtualized import V
from .common import IndentedBuffer
from .triton import TritonKernel


template_dict = {
    ir.Convolution: "triton_conv"
}
class TritonTemplateKernel(TritonKernel):
    def __init__(self, node: ir.ExternKernel, *groups):
        super(TritonTemplateKernel, self).__init__(*groups)
        self.node = node
        template_name = template_dict[type(node)]
        env = Environment(
            loader = FileSystemLoader("torchinductor/codegen"),
            trim_blocks = True,
            lstrip_blocks = True,
        )
        self.template = env.get_template(template_name+".j2")
        self.pointwise_compute = None

    def codegen_body(self, extra_argdefs):
        """
        get render_variables that to be put into the template
        to generate the final code
        """
        # TODO: codegen_body
        render_dict = {}
        render_dict["extra_args"] = extra_argdefs
        render_dict["pointwise_computation"] = self.pointwise_compute
        self.body = self.template.render(render_dict) + "\n"

    def codegen_kernel(self, name=None):

        code = IndentedBuffer()

        argdefs, _ = self.args.python_argdefs()

        # if config.dynamic_shapes:
        #     maybe_const = ""
        # else:
        #     maybe_const = ": tl.constexpr"

        # for tree in self.range_trees:
        #     if isinstance(tree.kernel, TritonKernel) \
        #         and (tree.prefix != "r" or self.inside_reduction):
        #         # skip current TritonTemplateKernel, no extra argdefs is needed
        #         argdefs.append(f"{tree.prefix}numel{maybe_const}")

        # for tree in self.range_trees:
        #     if isinstance(tree.kernel, TritonKernel) \
        #         and (tree.prefix != "r" or self.inside_reduction):
        #         argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        self.codegen_body(argdefs)
        code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")

        return wrapper.getvalue()

    def map_args(self, wrapper, kernel_name):
        """
        map the constant args or 
        kernel[grid](..., IN_C, IN_H, IN_W, strides,...)
        """
        map_dict, const_dict, other_dict = self.node.map_args()
        code = IndentedBuffer()
        # TODO: self.args = map_dict, const_dict
        return

    def precompute(self, wrapper, kernel_name):
        """
        some triton kernels needs host precompute tensor
        for example, triton_conv needs precompte delta_x_ptr
        """
        if  isinstance(self.node, ir.Convolution):
            wrapper.writeline("from torchinductor.triton_ops import _conv as _conv")
            wrapper.writeline(
                f"{kernel_name}_delta_x = _conv._delta_x_ptr(IN_C, KERNEL_H, KERNEL_W, dilation[0], dilation[1], stride_w[wc], stride_w[wh], stride_w[ww], stride_x[xc], stride_x[xh], stride_x[xw], device,)"
            )
        return
    
    def gen_grid(self):
        code = IndentedBuffer()
        with code.indent():
            code.splice(
                f"""
                def grid(META):
                    return (
                        triton.cdiv(BATCH * OUT_H * OUT_W, META["BLOCK_M"]),
                        triton.cdiv(KERNEL_N, META["BLOCK_N"]),
                    )
                """
            )
        return code.getvalue()

    def call_kernel(self, code, name: str):
        # gen code to call kernel
        # e.g. 
        # def grid(META):
        #     return (...)
        # kernel1[grid](arg0, arg1, ...)
        _, call_args = self.args.python_argdefs()
        call_args = ", ".join(call_args)
        code.writeline(self.gen_grid())
        code.writeline(f"{name}[grid]({call_args})")

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
    wrapper.header.splice(kernel.codegen_kernel(kernel_name))
    # map const args/ shape/ strides to kernel args
    kernel.map_args(wrapper, kernel_name)
    # gen precompute tensor (like delta_x_ptr) if needed
    kernel.precompute(wrapper, kernel_name)
    # code gen call to kernel
    kernel.call_kernel(wrapper, kernel_name)

    scheduler.enqueue(reschedule)  # TODO: consider reschedule
    scheduler.barrier()
    scheduler.maybe_free_buffers()
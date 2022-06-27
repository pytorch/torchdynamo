import os

from jinja2 import Environment
from jinja2 import FileSystemLoader

from .. import ir
from ..virtualized import V
from .common import IndentedBuffer
from .triton import TritonKernel

template_dict = {ir.Convolution: "triton_conv"}


class TritonTemplateKernel(TritonKernel):
    def __init__(self, node: ir.ExternKernel, *groups):
        super(TritonTemplateKernel, self).__init__(*groups)
        self.node = node
        template_name = template_dict[type(node)]
        env = Environment(
            loader=FileSystemLoader(os.path.dirname(__file__)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.template = env.get_template(template_name + ".j2")
        self.pointwise_compute = None

    def codegen_body(self, name, extra_argdefs):
        """
        get render_variables that to be put into the template
        to generate the final code
        """
        # TODO: codegen_body
        render_dict = {}
        render_dict["kernel_name"] = name
        render_dict["extra_args"] = extra_argdefs
        render_dict["pointwise_computation"] = self.pointwise_compute
        self.body = self.template.render(render_dict) + "\n"

    def codegen_kernel(self, name=None):

        code = IndentedBuffer()

        # self.args is the args for pointwise or reduction that will be fused
        # with the current TritonTemplateKernel
        extra_argdefs, _ = self.args.python_argdefs()

        self.codegen_body(name, extra_argdefs)
        code.splice(self.body)

        if name is not None:
            return code.getvalue()

        wrapper = IndentedBuffer()
        wrapper.writeline("TritonCodeCache.load('''")
        wrapper.splice(code.getvalue(), strip=True)
        wrapper.writeline("''').kernel")

        return wrapper.getvalue()

    def map_args(self):
        """
        map the constant args or
        kernel[grid](..., IN_C, IN_H, IN_W, strides,...)
        """
        self.args_dict, self.const_dict, self.other_dict = self.node.map_args()

    def precompute(self, wrapper, kernel_name):
        """
        some triton kernels needs host precompute tensor
        for example, triton_conv needs precompte delta_x_ptr
        """
        if isinstance(self.node, ir.Convolution):
            if self.const_dict["CONV1X1_NHWC"] == "False":
                self.args_dict["delta_x_ptr"] = "delta_x"
                wrapper.writeline("from torchinductor.triton_ops import _conv as _conv")
                IN_C = self.args_dict["IN_C"]
                KERNEL_H = self.args_dict["KERNEL_H"]
                KERNEL_W = self.args_dict["KERNEL_W"]
                dilation_h = self.args_dict["dilation_h"]
                dilation_w = self.args_dict["dilation_w"]
                stride_wc = self.args_dict["stride_wc"]
                stride_wh = self.args_dict["stride_wh"]
                stride_ww = self.args_dict["stride_ww"]
                stride_xc = self.args_dict["stride_xc"]
                stride_xh = self.args_dict["stride_xh"]
                stride_xw = self.args_dict["stride_xw"]
                device = self.other_dict["device"]
                wrapper.writeline(
                    "delta_x = _conv._delta_x_ptr("
                    f"{IN_C}, {KERNEL_H}, {KERNEL_W}, "
                    f"{dilation_h}, {dilation_w}, "
                    f"{stride_wc}, {stride_wh}, {stride_ww}, "
                    f"{stride_xc}, {stride_xh}, {stride_xw}, {device})"
                )
            # else, delta_x_ptr is None
        return

    def gen_grid(self, name):
        code = IndentedBuffer()
        if isinstance(self.node, ir.Convolution):
            BATCH = self.args_dict["BATCH"]
            OUT_H = self.args_dict["OUT_H"]
            OUT_W = self.args_dict["OUT_W"]
            KERNEL_N = self.args_dict["KERNEL_N"]
            with code.indent():
                code.splice(
                    f"""
                    def grid_{name}(META):
                        return (
                            triton.cdiv({BATCH} * {OUT_H} * {OUT_W}, META["BLOCK_M"]),
                            triton.cdiv({KERNEL_N}, META["BLOCK_N"]),
                        )
                    """
                )
        return code.getvalue()

    def call_kernel(self, wrapper, name: str):
        # gen code to call kernel
        # e.g.
        # def grid(META):
        #     return (...)
        # kernel1[grid](arg0, arg1, ...)
        extra_arg_defs, extra_call_args = self.args.python_argdefs()
        extra_args = ", ".join(extra_call_args)
        self_args = ", ".join({**self.args_dict, **self.const_dict}.values())
        args = self_args + (
            ", " + extra_args if extra_args and len(extra_args) > 0 else ""
        )
        wrapper.writeline(self.gen_grid(name))
        wrapper.writeline(f"{name}[grid_{name}]({args})")


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
    kernel.map_args()
    # gen precompute tensor (like delta_x_ptr) if needed
    kernel.precompute(wrapper, kernel_name)
    # code gen call to kernel
    kernel.call_kernel(wrapper, kernel_name)

    scheduler.enqueue(reschedule)  # TODO: consider reschedule
    scheduler.barrier()
    scheduler.maybe_free_buffers()

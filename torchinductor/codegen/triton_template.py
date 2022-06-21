import os

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import StrictUndefined

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
            undefined=StrictUndefined,
        )
        self.template = env.get_template(template_name + ".j2")
        # self.pointwise_compute = []
        # self.triton_kernels = []
        # self.triton_kernel_indexing_code = []
        # self.triton_kernel_loads = []
        # self.triton_kernel_compute = []
        # self.triton_kernel_stores = []

    # def add_fusable_node(self, node: TritonKernel):
    #     self.triton_kernels.append(node)
    #     self.triton_kernel_indexing_code.append(node.indexing_code)
    #     self.triton_kernel_loads.append(node.loads)
    #     self.triton_kernel_compute.append(node.compute)
    #     self.triton_kernel_stores(node.stores)

    def codegen_body(self, name):
        """
        get render_variables that to be put into the template
        to generate the final code
        """
        # get extra_argdefs from fusable triton kernels
        self.extra_argdefs = []
        self.extra_call_args = []
        argdefs, call_args = self.args.python_argdefs()
        # add extra args if it is different from 
        # current TritonTemplateKernel args
        for (argdef, call_arg) in zip(argdefs, call_args):
            if call_arg not in self.node.codegen_args():
                self.extra_argdefs.append(argdef)
                self.extra_call_args.append(call_arg)

        self.pointwise_code = IndentedBuffer()
        self.pointwise_code.splice(self.indexing_code)
        self.pointwise_code.splice(self.loads)
        self.pointwise_code.splice(self.compute)
        self.pointwise_code.splice(self.stores)
        render_dict = {}
        render_dict["kernel_name"] = name
        render_dict["extra_argdefs"] = self.extra_argdefs
        render_dict["pointwise_code"] = self.pointwise_code.getvalue()
        self.body = self.template.render(render_dict) + "\n"

    def codegen_kernel(self, name=None):

        code = IndentedBuffer()

        self.codegen_body(name)
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

    def gen_grid(self):
        code = IndentedBuffer()
        if isinstance(self.node, ir.Convolution):
            BATCH = self.args_dict["BATCH"]
            OUT_H = self.args_dict["OUT_H"]
            OUT_W = self.args_dict["OUT_W"]
            KERNEL_N = self.args_dict["KERNEL_N"]
            with code.indent():
                code.splice(
                    f"""
                    def grid(META):
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
        extra_args = ", ".join(self.extra_call_args)
        self_args = ", ".join({**self.args_dict, **self.const_dict}.values())
        args = self_args + (
            ", " + extra_args if extra_args and len(extra_args) > 0 else ""
        )
        wrapper.writeline(self.gen_grid())
        wrapper.writeline(f"{name}[grid]({args})")


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
        # mark node of TritonTemplateKernel as fusable and update fusable_deps
        node.mark_fusable()
        # enqueue any nodes that became runable after this node is run
        # otherwise, relu after conv is in blocked_nodes that could not be in
        # runable_groups to be fused to conv
        scheduler.barrier()
        # scheduler.pop_group will keep iterating all reachable fusable SchedulerNodes
        for node in scheduler.pop_group(group):
            # scheduler.maybe_remove_buffer(node, check_group=is_group_matching)
            node.run(*kernel.set_ranges(*node.get_ranges()))
            node.mark_fusable()
            # collect loads/ stores/ compute inside 
            # kernel.add_fusable_node(node.node)

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
        scheduler.barrier() # enqueue any nodes that became runable after this node is run
        scheduler.maybe_free_buffers()

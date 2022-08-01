import logging

import torch

from torchdynamo import config
from torchdynamo.utils import clone_inputs
from torchdynamo.utils import count_calls
from torchdynamo.utils import counters

from .analysis import has_mutation
from .backends import BACKENDS
from .normalize import normalize_ir

log = logging.getLogger(__name__)


class AotAutogradStrategy(object):
    """Base class for backend strategies that use AOT Autograd"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm.forward  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        import functorch.compile

        super(AotAutogradStrategy, self).__init__()
        counters["aot_autograd"]["total"] += 1
        self.use_fallback = False
        self.original_example_inputs = example_inputs
        self.gm = gm

        if not functorch.compile.config.use_functionalize and config.normalize_ir:
            try:
                self.gm = normalize_ir(gm, self.example_inputs)
            except Exception:
                log.debug("TorchDynamo unable to remove mutation")
                self.use_fallback = True
                pass

        gm_inputs = list(filter(lambda x: x.op == "placeholder", gm.graph.nodes))

        # 1) LSTM module (tts_angular) - https://github.com/pytorch/functorch/issues/586
        for submod in self.gm.modules():
            if submod.__class__.__name__ == "LSTM":
                self.use_fallback = True

        # 2) set_grad_enabled
        has_set_grad_enabled = False
        for node in self.gm.graph.nodes:
            if node.target == torch._C._set_grad_enabled:
                has_set_grad_enabled = True

        if functorch.compile.config.use_functionalize:
            # There are two problematic classes we still exclude for now with
            # functionalization:
            #   - data mutation of inputs (fixed when we stop recording the
            #   copy_ directly into the graph)
            #   - metadata mutation of inputs (fixed if we do an extra partition
            #   to avoid AOTAutograd on the mutated inputs, or if we some how
            #   get custom autograd function to reflect metadata changes to the
            #   original tensor)
            mutated = has_mutation(self.gm, self.example_inputs, inputs_only=True)
        else:
            mutated = has_mutation(self.gm, self.example_inputs)

        if mutated or len(gm_inputs) == 0 or has_set_grad_enabled:
            self.use_fallback = True

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        if self.use_fallback:
            log.debug("Unable to use AOT Autograd because graph has mutation")
            counters["aot_autograd"]["not_ok"] += 1
            return self.gm
        cg = self.candidate()
        if cg is None:
            counters["aot_autograd"]["not_ok"] += 1
            raise RuntimeError("AOT Autograd failed to compile")
        counters["aot_autograd"]["ok"] += 1
        return cg

    def candidate(self):
        raise NotImplementedError()


class AotNop(AotAutogradStrategy):
    """Useful for debugging purpose"""

    def candidate(self):
        from functorch.compile import nop

        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, fw_compiler=nop)


aot_nop = AotNop.compile_fn


class AotTorchscript(AotAutogradStrategy):
    """
    AOT Autograd with torchscript backend. Default partitioner.
    """

    def candidate(self):
        from functorch.compile import ts_compile

        return BACKENDS["aot_autograd"](
            self.gm, self.example_inputs, fw_compiler=ts_compile
        )


aot_ts = AotTorchscript.compile_fn

# Global counter to differentiate between different graphs.
graph_idx = 0


class AotPrint(AotNop):
    """Saves all the gm models so that we can run them separately"""

    def candidate(self):
        global graph_idx
        module_idx = "module_" + str(graph_idx)
        self.gm.to_folder(module_idx, "Bar")
        for idx, x in enumerate(self.example_inputs):
            torch.save(x, module_idx + "_tensor" + str(idx) + ".pt")
        graph_idx += 1
        return super(AotPrint, self).candidate()


aot_print = AotPrint.compile_fn


def mem_efficient_fusion_kwargs(use_decomps):
    from functorch.compile import default_decompositions
    from functorch.compile import min_cut_rematerialization_partition
    from functorch.compile import ts_compile

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
        "hasher_type": "StaticShapeHasher",
    }

    if use_decomps:
        kwargs["decompositions"] = default_decompositions

    return kwargs


class AotMemEfficientFusion(AotAutogradStrategy):
    """Use Min cut rematerilization and NVFuser with AOT Autograd"""

    def candidate(self):
        kwargs = mem_efficient_fusion_kwargs(use_decomps=True)
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, **kwargs)


class AotMemEfficientFusionNoDecomps(AotAutogradStrategy):
    """Use Min cut rematerilization and NVFuser with AOT Autograd"""

    def candidate(self):
        kwargs = mem_efficient_fusion_kwargs(use_decomps=False)
        return BACKENDS["aot_autograd"](self.gm, self.example_inputs, **kwargs)


class AOTMemEfficientFusionWithContext:
    """Pass nvfuser context to TorchDynamo"""

    def __init__(self, use_decomps=True):
        self.backend_ctx_ctor = lambda: torch.jit.fuser("fuser2")
        self.use_decomps = use_decomps

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        if self.use_decomps:
            return AotMemEfficientFusion.compile_fn(gm, example_inputs)
        else:
            return AotMemEfficientFusionNoDecomps.compile_fn(gm, example_inputs)


aot_mem_efficient_fusion = AOTMemEfficientFusionWithContext(True)
aot_mem_efficient_fusion_no_decomp = AOTMemEfficientFusionWithContext(False)


class AotPrimsNvfuser(AotAutogradStrategy):
    """
    Use FX graph partitioner + Aten2Prims ref + trace executor + nvFuser
    """

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        super(AotPrimsNvfuser, self).__init__(gm, example_inputs)

        from functorch.compile import min_cut_rematerialization_partition
        from torch.fx.passes.backends.nvfuser import NvFuserBackend

        self.nvfuser = NvFuserBackend()
        self.min_cut_rematerialization_partition = min_cut_rematerialization_partition
        self.populate_aten2aten_decomps()

    def populate_aten2aten_decomps(self):
        from torch._decomp import get_decompositions

        aten = torch.ops.aten
        default_decompositions = {
            aten.detach,
            aten.gelu_backward,
            aten.leaky_relu_backward,
            aten.sigmoid_backward,
            aten.threshold_backward,
            aten.hardtanh_backward,
            aten.hardsigmoid_backward,
            aten.hardswish_backward,
            aten.tanh_backward,
            aten.silu_backward,
            aten.elu_backward,
            aten.cudnn_batch_norm,
            aten.cudnn_batch_norm_backward,
            aten.masked_fill.Scalar,
            aten.masked_fill.Tensor,
            aten.elu,
            aten.leaky_relu,
            aten.hardtanh,
            aten.hardswish,
            aten.hardsigmoid,
            aten.rsub,
            aten.native_batch_norm_backward,
        }

        self.aten2aten_decompositions = get_decompositions(default_decompositions)

    def candidate(self):
        return BACKENDS["aot_autograd"](
            self.gm,
            self.example_inputs,
            fw_compiler=self.nvfuser,
            partition_fn=self.min_cut_rematerialization_partition,
            hasher_type="StaticShapeHasher",
            decompositions=self.aten2aten_decompositions,
        )


aot_prims_nvfuser = AotPrimsNvfuser.compile_fn


def create_aot_backends():
    """
    Register aliases for the AOT backends
    """
    # aot_nop uses AOT Autograd backend with nop compiler. It is helpful in debugging.
    BACKENDS["aot_nop"] = aot_nop

    # aot_nop uses AOT Autograd backend with print compiler. It prints the
    # graphs and also saves the graph modules that are sent to AOT Autograd.
    # This is helpful for debugging.
    BACKENDS["aot_print"] = aot_print

    # aot_ts uses torchscript backend. We can use this with both nnc and nvfuser
    # by using the relevant fuser with torch.jit.fuser(...)
    BACKENDS["aot_ts"] = aot_ts

    # prims_nvfuser uses the prims and AOT-Autograd to get FX-aten IR. And then
    # directly lowers to NVFuser without relying no Torchscript.
    BACKENDS["prims_nvfuser"] = aot_prims_nvfuser

    # aot_nvfuser uses the memory efficient fusion algorithm from AOT Autograd.
    # It uses min cut rematerialization algorithm, and uses nvfuser as the
    # compiler backend. This is the most optimized setting with nvfuser for
    # training.
    BACKENDS["aot_nvfuser"] = aot_mem_efficient_fusion

    # Similar to aot_nvfuser, but disables the decompositions. Decompositions
    # can cause accuracy deviations. This setting allows us to compare accuracy
    # without worrying about the impact of decomposisitons. More details at
    # https://github.com/pytorch/torchdynamo/issues/611
    BACKENDS["aot_nvfuser_nodecomps"] = aot_mem_efficient_fusion_no_decomp

import logging

import torch

from torchdynamo.utils import clone_inputs
from torchdynamo.utils import count_calls

from .analysis import has_mutation
from .backends import BACKENDS
from .normalize import normalize_ir

log = logging.getLogger(__name__)


class AOTAutogradStrategy(object):
    """Base class for backend strategies that use AOT Autograd"""

    @classmethod
    def compile_fn(cls, gm: torch.fx.GraphModule, example_inputs):
        if count_calls(gm.graph) < 2:
            return gm.forward  # no point for tiny graphs
        return cls(gm, example_inputs).verified_candidate()

    def __init__(self, gm: torch.fx.GraphModule, example_inputs):
        super(AOTAutogradStrategy, self).__init__()
        # TODO - Look why restore fails with train() even after restoring.
        # self.restore = checkpoint_params(gm)
        # self.correct = gm.forward(*self.example_inputs)
        self.original_example_inputs = example_inputs
        self.gm = normalize_ir(gm, self.example_inputs)
        self.use_fallback = False
        gm_inputs = list(filter(lambda x: x.op == "placeholder", gm.graph.nodes))

        if has_mutation(self.gm, self.example_inputs) or len(gm_inputs) == 0:
            self.use_fallback = True

    @property
    def example_inputs(self):
        return clone_inputs(self.original_example_inputs)

    def verified_candidate(self):
        if self.use_fallback:
            log.warn("Unable to use AOT Autograd because graph has mutation")
            return self.gm
        return self.candidate()

    def candidate(self):
        raise NotImplementedError()


class AOTAutogradEagerStrategy(AOTAutogradStrategy):
    """Useful for debugging purpose"""

    def candidate(self):
        from functorch.compile import print_compile

        cg = BACKENDS["aot_autograd"](
            self.gm, self.example_inputs, fw_compiler=print_compile
        )
        if cg is None:
            raise RuntimeError("AOT Autograd failed to compile")
        return cg


aot_autograd_debug_strategy1 = AOTAutogradEagerStrategy.compile_fn

# Global counter to differentiate between different graphs.
graph_idx = 0


class AOTAutogradEagerSaveStrategy(AOTAutogradEagerStrategy):
    """Saves all the gm models so that we can run them separately"""

    def candidate(self):
        global graph_idx
        module_idx = "module_" + str(graph_idx)
        self.gm.to_folder(module_idx, "Bar")
        for idx, x in enumerate(self.example_inputs):
            torch.save(x, module_idx + "_tensor" + str(idx) + ".pt")
        graph_idx += 1
        return super(AOTAutogradEagerSaveStrategy, self).candidate()


aot_autograd_debug_strategy2 = AOTAutogradEagerSaveStrategy.compile_fn


class AOTAutogradMemoryEfficientFusion(AOTAutogradStrategy):
    """Use Min cut rematerilization and NVFuser with AOT Autograd"""

    def candidate(self):
        cg = BACKENDS["aot_autograd"](self.gm, self.example_inputs)
        if cg is None:
            raise RuntimeError("AOT Autograd failed to compile")
        return cg


aot_autograd_speedup_strategy = AOTAutogradMemoryEfficientFusion.compile_fn

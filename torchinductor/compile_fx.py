from typing import List

import torch.fx

from torchdynamo.optimizations.python_key import python_key_normalize
from torchinductor.graph import GraphLowering

from . import config
from . import virtualized


def compile_fx(model: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """Main entrypoint to a compile given FX graph"""
    assert isinstance(model, torch.fx.GraphModule)
    assert all(isinstance(x, torch.Tensor) for x in example_inputs)
    gm, wrap = python_key_normalize(model, example_inputs)
    if config.dce:
        gm.graph.eliminate_dead_code()
    graph = GraphLowering(gm)
    with virtualized.graph.set_handler(graph):
        wrap(graph.run)(*example_inputs)
        return wrap(graph.compile_to_fn())

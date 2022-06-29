import itertools
from typing import Dict
from typing import List

import torch
import torch.fx as fx


class DDPOptimizer:
    def __init__(
        self, bucket_bytes_cap: int, parameters_to_ignore: List[str], backend_compile_fn, debug=False
    ):
        self.bucket_bytes_cap = bucket_bytes_cap
        self.parameters_to_ignore = parameters_to_ignore
        self.backend_compile_fn = backend_compile_fn
        self.debug = debug

    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        TODO:
        - handle params_and_buffers_to_ignore
        - handle kwargs
        """

        # 1: compute the partition map according to DDP bucket logic
        bucket_bytes = 0
        bucket_actual_sizes = []
        node_splits = [[]]
        for node in reversed(gm.graph.nodes):
            if bucket_bytes >= self.bucket_bytes_cap:
                bucket_actual_sizes.insert(0, bucket_bytes)
                bucket_bytes = 0
                node_splits.insert(0, [])

            if node.op == "output" or node.op == "placeholder":
                continue

            elif node.op == "call_module":
                target = gm.get_submodule(node.target)
                params_size_b = sum(
                    [
                        p.storage().nbytes()
                        for p in target.parameters()
                        if p.requires_grad
                    ]
                )
                bucket_bytes += params_size_b
                print(f"accumulated {params_size_b} b from {node}")
            else:
                # TODO(whc) confirm this:
                # e.g. call_method, call_function aren't supported, as they wouldn't be supported by DDP either.
                pass

            node_splits[0].append(node)

        if len(node_splits) == 1:
            if self.debug:
                print(
                    f"DDPOptimizer did not split graphs. Accumulated {bucket_bytes} bytes, and bucket cap is {self.bucket_bytes_cap}"
                )
            return gm

        if len(bucket_actual_sizes) < len(node_splits):
            bucket_actual_sizes.insert(0, bucket_bytes)

        if self.debug:
            print(
                f"DDPOptimizer used bucket cap {self.bucket_bytes_cap} and split graphs into parameter sizes {', '.join([str(b) for b in bucket_actual_sizes])}"
            )

        # 2: partition the graphmodule according to bucket capacity
        partition_map = {}
        for p, nodes in enumerate(node_splits):
            for node in nodes:
                partition_map[node] = p
        split_gm = fx.passes.split_module.split_module(
            gm, None, lambda node: partition_map[node]
        )

        # 3: compile each of the partitioned submodules using the user-provided compiler
        new_graph = fx.Graph()
        val_map: Dict[fx.Node, fx.Node] = {}
        subgraph_example_inputs = example_inputs
        for node in split_gm.graph.nodes:
            if node.op == "call_module":

                # submods may return a single tensor or a tuple
                # but AotAutograd requires even single-tensors to be wrapped in tuples
                # we have to wrap/unwrap the special case singleton-tuples using this wrapper
                submod = split_gm.get_submodule(node.target)
                unwrap_singleton_tuple = False
                for sn in submod.graph.nodes:
                    if sn.op == "output":
                        if not isinstance(sn.args[0], tuple):
                            unwrap_singleton_tuple = True
                            sn.args = (sn.args,)

                class WrapperModule(torch.nn.Module):
                    def __init__(self, compiled_submod, unwrap_singleton_tuple):
                        super().__init__()
                        self.compiled_submod = compiled_submod
                        self.unwrap_singleton_tuple = unwrap_singleton_tuple

                    def forward(self, *args):
                        if self.unwrap_singleton_tuple:
                            return self.compiled_submod(*args)[0]
                        return self.compiled_submod(*args)

                wrapper = WrapperModule(
                    self.backend_compile_fn(submod, subgraph_example_inputs),
                    unwrap_singleton_tuple,
                )

                # propagate the example inputs for use when compiling next submod
                subgraph_example_inputs = wrapper.compiled_submod(
                    *subgraph_example_inputs
                )

                # replace the original submod with the compiled one in its wrapper
                split_gm.delete_submodule(node.target)
                split_gm.add_submodule(node.target, wrapper,
                )

            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
        split_gm.graph = new_graph

        if self.debug:
            print("DDPOptimizer compiled the split graphs:")
            print(split_gm.graph)
            print()

        return split_gm

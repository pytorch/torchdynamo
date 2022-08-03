import torch.fx as fx
from typing import List
import torch
from contextlib import contextmanager

class GraphProducer(object):
    def __init__(self):
      self.possible_input_names: set[str] = set()
      self.inputs: list[str] = []  # this is used to iterate the above set given it's iteration is abitrary.
      self.code: list[fx.Node] = []
      self.possible_output_names: set[str] = set()
      self.outputs: list[fx.Node] = []
      self._graph: fx.Graph = None

    def graph(self):
      if self._graph is not None:
        return self._graph

      self._graph = fx.Graph()
      for input in self.inputs:
        self._graph.placeholder(input)
      for code in self.code:
        self._graph.node_copy(code)
      self._graph.output(tuple(self.outputs))
      return self._graph

class DDPOptimizer:
    def __init__(self, bucket_cap_mb: int, parameters_to_ignore: List[str], backend_compile_fn):
        self.bucket_cap_mb = bucket_cap_mb
        self.parameters_to_ignore = parameters_to_ignore
        self.backend_compile_fn = backend_compile_fn
    
    def compile_fn(self, gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
        """
        TODO:
        - handle params_and_buffers_to_ignore
        - implement call_method/call_forward
        - handle kwargs
        """

        print("DDPOptimizer called with FX graph:")
        gm.graph.print_tabular()
        print()
        # 1. Splitting the gm into small graphs, following DDP parameter bucketing
        named_params = {name: param for name, param in gm.named_parameters()}
        bucket_cap_b = self.bucket_cap_mb * 10**6
        bucket_bytes = 0
        graphs = [GraphProducer()]
        bucket_actual_sizes = []
        # Init the first node to use the output of the full graph if appropriate.
        # Somehow the args for the output node is a tupel of args.
        graphs[0].possible_output_names = {str(arg) for arg in list(gm.graph.nodes)[-1].args[0]}
        for node in reversed(gm.graph.nodes):
            # To be noted, we don't clear the outputs from the last graph
            # given that's the arguments for the output node too.
            if bucket_bytes >= bucket_cap_b:
                assert len(graphs) > 0
                graphs[0].inputs = list(graphs[0].possible_input_names)
                bucket_actual_sizes.insert(0, bucket_bytes)
                bucket_bytes = 0
                graphs.insert(0, GraphProducer())

                # Set the possible output of the new graph
                # For any remaining possible outputs in the next graph, it could
                # either be produced by the current graph or be the input for the
                # full graph. Therefore, carry them on.
                graphs[0].possible_output_names = graphs[1].possible_output_names
                for input in graphs[1].inputs:
                    graphs[0].possible_output_names.add(input)

            if node.op == "output" or node.op == "placeholder":
                continue

            elif node.op == "call_module":
                target = gm.get_submodule(node.target)
                params_size_b = sum([p.storage().nbytes() for p in target.parameters() if p.requires_grad])
                bucket_bytes += params_size_b
                print(f"accumulated {params_size_b} b from {node}")
            else:
                # e.g. call_method, call_function: should be easy to support but do it along with tests
                assert False, f"TODO: {node.op} is not yet supported"
            
            graphs[0].code.insert(0, node)
            if node.name in graphs[0].possible_output_names:
                graphs[0].outputs.append(node)
                graphs[0].possible_output_names.remove(node.name)

            if node.name in graphs[0].possible_input_names:
                graphs[0].possible_input_names.remove(node.name)

            # TODO: Deal with default arguments?
            for arg in node.args:
                graphs[0].possible_input_names.add(str(arg))
            # TODO: Do we care the kwargs?
            # for key, value in node.kwargs.items():
            #   outputs.add(value)
            graphs[0].inputs = list(graphs[0].possible_input_names)

        if len(graphs) == 1:
            print(f"DDPOptimizer did not split graphs. Accumulated {bucket_bytes} bytes, and bucket cap is {bucket_cap_b}")
            return gm

        print(f"DDPOptimizer used bucket cap {bucket_cap_b} and split graphs into {','.join([str(b) for b in bucket_actual_sizes])}")
        for graph in graphs:
            graph.graph().print_tabular()
        print()

        # 2. Compiling the splitted graphs using the provided user compiler
        gms = [fx.GraphModule(gm, graph.graph()) for graph in graphs]
        compiled_submods = []
        for g in gms:
            compiled = self.backend_compile_fn(g, None)
            assert compiled is not None, "aot compilation failed"
            compiled_submods.append(compiled)

        print(f"DDPOptimizer compiled all {len(compiled_submods)} modules\n")

        # 3. Stitching the compiled graphs back to a fx gm to return.
        assert len(compiled_submods) == len(graphs)
        final_graph = fx.Graph()
        arg_map = {}  # To keep track of all possible inputs to sub-graphs.

        for node in gm.graph.nodes:
            if node.op != "placeholder":
                break
            last_node = final_graph.node_copy(node)
            arg_map[last_node.name] = last_node
        
        # We call the AOT compiled sub module with arg in arg_map that match recorded arg name in the corresponding graph.
        for i in range(len(graphs)):
            output = final_graph.call_function(compiled_submods[i], tuple([arg_map[input] for input in graphs[i].inputs]))
            # Unpack the output and name them properly so that they can be fetched correctly when needed in consecutive graphs.
            for j in range(len(graphs[i].outputs)):
                getitem = final_graph.call_method("__getitem__", (output, j))
                getitem.name = graphs[i].outputs[j].name
                arg_map[getitem.name] = getitem

        final_graph.node_copy(list(gm.graph.nodes)[-1])
        final_graph_module = fx.GraphModule(gm, final_graph)

        print("DDPOptimizer joined split graphs:")
        final_graph_module.graph.print_tabular()
        print()

        return final_graph_module.forward  # return a python callable

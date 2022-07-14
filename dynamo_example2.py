import os
import torch
import torch.fx as fx
import torch.nn as nn
import torch.optim as optim

from typing import List

import torchdynamo
from torchdynamo.optimizations import BACKENDS

from torch.profiler import profile, record_function, ProfilerActivity

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10000)
        self.net2 = nn.Linear(10000, 10000)
        self.net3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()
        self.net4 = nn.Linear(10000, 5)

    def forward(self, x):
        return self.net4(self.relu(self.net3(self.relu(self.net2(self.relu(self.net1(x)))))))

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

# A very rough PoC that split the input fx gragh into small graphs, and then
# stitch the compiled aot graphs back to one.
def graph_break_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("graph_break_compiler() called with FX graph:")
    gm.graph.print_tabular()
    print()

    # 1. Splitting the gm into small graphs. Currently, we just naively split
    # the graph every 3 ops.
    magic_number = 3  # magic number to break the graph
    count = 0
    graphs = [GraphProducer()]
    # Init the first node to use the output of the full graph if appropriate.
    # Somehow the args for the output node is a tupel of args.
    graphs[0].possible_output_names = {str(arg) for arg in list(gm.graph.nodes)[-1].args[0]}
    for node in reversed(gm.graph.nodes):
      # To be noted, we don't clear the outputs from the last graph
      # given that's the arguments for the output node too.
      if count == magic_number:
        assert len(graphs) > 0
        graphs[0].inputs = list(graphs[0].possible_input_names)
        count = 0
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

      count = count + 1
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

    print("graph_break_compiler() called with splitted graphs:")
    for graph in graphs:
      graph.graph().print_tabular()
      print()

    # 2. Compiling the splitted graphs using AOT.
    gms = [fx.GraphModule(gm, graph.graph()) for graph in graphs]
    aot_compileds = []
    for g in gms:
      aot_compiled = BACKENDS["aot_autograd"](g, None)
      assert aot_compiled is not None, "aot compilation failed"
      aot_compileds.append(aot_compiled)

    print(f"AOT compiled all {len(aot_compileds)} modules\n")

    # 3. Stitching the compiled graphs back to a fx gm to return.
    assert len(aot_compileds) == len(graphs)
    final_graph = fx.Graph()
    arg_map = {}  # To keep track of all possible inputs to sub-graphs.

    for node in gm.graph.nodes:
      if node.op != "placeholder":
        break
      last_node = final_graph.node_copy(node)
      arg_map[last_node.name] = last_node

    # We call the AOT compiled sub module with arg in arg_map that match recorded arg name in the corresponding graph.
    for i in range(len(graphs)):
      output = final_graph.call_function(aot_compileds[i].forward, tuple([arg_map[input] for input in graphs[i].inputs]))
      # Unpack the output and name them properly so that they can be fetched correctly when needed in consecutive graphs.
      for j in range(len(graphs[i].outputs)):
        getitem = final_graph.call_method("__getitem__", (output, j))
        getitem.name = graphs[i].outputs[j].name
        arg_map[getitem.name] = getitem

    final_graph.node_copy(list(gm.graph.nodes)[-1])
    final_graph_module = fx.GraphModule(gm, final_graph)

    print("graph_break_compiler() called with stitched graph:")
    final_graph_module.graph.print_tabular()
    print()

    return final_graph_module.forward  # return a python callable

def hook(grad):
  print("gradient hook fired")
  grad + 1
  return grad

def demo_basic():
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
      with torchdynamo.optimize(graph_break_compiler):
        device = "cuda"
        # device = "cpu"

        # create model and move it to the device with id rank
        model = ToyModel().to(device)
        for parameter in model.parameters():
          parameter.register_hook(hook)

        loss_fn = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001)

        for i in range(1):
            optimizer.zero_grad()
            outputs = model(torch.randn(20, 10).to(device))
            labels = torch.randn(20, 5).to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            print(f"{os.getpid()}: iteration {i}, loss {loss}")

    prof.export_chrome_trace("new_trace.json")

if __name__ == "__main__":
    demo_basic()

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

# A very rough PoC that split the input fx gragh into small graphs, and then
# stitch the compiled aot graphs back to one.
def graph_break_compiler(gm: fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("graph_break_compiler() called with FX graph:")
    gm.graph.print_tabular()
    print()

    def insert_node(graph: fx.Graph, node: fx.Node):
      if len(graph.nodes) == 0:
        graph.create_node(node.op, node.target, node.args, node.kwargs, node.name, node.type)
      else:
        with graph.inserting_before(list(graph.nodes)[0]):
          graph.node_copy(node)

    graphs = list()
    magic_number = 3  # magic number to break the graph
    count = magic_number
    outputs = {}
    for node in reversed(gm.graph.nodes):
      # To be noted, we don't clear the outputs from the last graph
      # given that's the arguments for the output node too.
      if count == magic_number:
        count = 0
        graphs.insert(0, fx.Graph())
        if len(graphs) > 1:
          # set the output of the new graph
          graphs[0].output(tuple(value for _, value in outputs.items()))
          # set the input of the previous graph
          # reverse the order to make the order match the above output
          for key, _ in reversed(outputs.items()):
            with graphs[1].inserting_before(list(graphs[1].nodes)[0]):
              graphs[1].placeholder(key)

      if node.op != "output" and node.op != "placeholder":
        count = count + 1

      insert_node(graphs[0], node)
      if node.name in outputs:
        outputs.pop(node.name)
      # TODO: Deal with default arguments?
      for arg in node.args:
        # Somehow the output node's args is a tuple of tuple.
        if type(arg) is tuple:
          for a in arg:
            outputs[str(a)] = a
          continue
        outputs[str(arg)] = arg
      # TODO: Do we care the kwargs?
      # for key, value in node.kwargs.items():
      #   outputs.add(value)

    print("graph_break_compiler() called with splitted graphs:")
    for graph in graphs:
      graph.print_tabular()
      print()

    gms = [fx.GraphModule(gm, graph) for graph in graphs]
    aot_compileds = []
    for g in gms:
      aot_compiled = BACKENDS["aot_autograd"](g, None)
      assert aot_compiled is not None, "aot compilation failed"
      aot_compileds.append(aot_compiled)

    print(f"AOT compiled all {len(aot_compileds)} modules\n")

    assert len(aot_compileds) == len(graphs)
    final_graph = fx.Graph()
    last_aot = None
    for i in range(len(aot_compileds)):
      arguments = list()
      j = 0
      for node in graphs[i].nodes:
        if node.op != "placeholder":
          break
        if i == 0:
          last_node = final_graph.node_copy(node)
        else:
          assert last_aot is not None
          last_node = final_graph.call_method("__getitem__", (last_aot, j))
        j = j + 1
        arguments.append(last_node)

      last_aot = final_graph.call_function(aot_compileds[i].forward, tuple(arguments))
    final_graph.output(last_aot)
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

    prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    demo_basic()

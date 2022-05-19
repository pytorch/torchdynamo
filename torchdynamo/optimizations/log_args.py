import json

import torch

from torchdynamo.optimizations.python_key import python_key_normalize

aten = torch.ops.aten


class ConvArgsAnalysis(torch.fx.Interpreter):
    """
    Log arguments like input shape (input, bias, weights shape)
    and options(padding/stride/kernel size/dilation/etc) for
    aten.convolution
    """

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)

        self.nodes_conv_args = {}
        # self.convolution_args = inspect.getfullargspec(aten.convolution.op())[0]
        # print(self.convolution_args)
        # self.convolution_nargs = len(self.convolution_args)

    def run(self, *args):
        run_result = super().run(*args)
        with open("tmp/conv_args.json", "w") as fd:
            json.dump(self.nodes_conv_args, fd)
        return run_result

    def run_node(self, n: torch.fx.Node):
        result = super().run_node(n)

        if n.op == "call_function":
            if n.target == aten.convolution:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                assert len(args) == 9, "aten.convolution should have 9 args"
                conv_args = {}
                conv_args["input"] = args[0].shape
                conv_args["weight"] = args[1].shape
                conv_args["bias"] = args[2].shape if args[2] is not None else None
                conv_args["stride"] = args[3]
                conv_args["padding"] = args[4]
                conv_args["dilation"] = args[5]
                conv_args["transposed"] = args[6]
                conv_args["output_padding"] = args[7]
                conv_args["groups"] = args[8]

                self.nodes_conv_args[n.name] = conv_args
        return result


def conv_args_analysis(gm: torch.fx.GraphModule, example_inputs):
    # lowering graph
    gm, wrap = python_key_normalize(gm, example_inputs)
    # use Interpreter to logs the args of conv
    wrap(ConvArgsAnalysis(gm).run)(*example_inputs)
    return wrap(gm.forward)

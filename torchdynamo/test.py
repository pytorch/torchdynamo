import torch
import torch.fx
import torch.nn.functional as F
from typing import Callable
import torchdynamo
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import default_qconfig

class Submodule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)

    def forward(self, x):
        return F.relu(self.linear1(x)) * self.scale


class BasicModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.scale = torch.randn(1, 10)
        self.sub = Submodule()

    def forward(self, x):
        x = F.relu(self.linear1(x)) * self.scale
        x = self.sub(x)
        return x


class QuantizationWrapper:
    def __init__(self, quant_compiler, model, example_inputs):
        super().__init__()
        self.quant_compiler = quant_compiler
        self.model = model
        self.prepare_mode = True
        self.prepared_model = None
        self.quantized_model = None

    def __call__(self, *args, **kwargs):
        print("prepare mode in QuantizationWrapper:", self.prepare_mode)
        if self.quant_compiler.prepare_mode:
            print("running prepared model")
            if self.prepared_model is None:
                self.model.eval()
                self.prepared_model = prepare_fx(self.model, self.quant_compiler.prepare_qconfig_dict)
                print("prepared model:", self.prepared_model)
            return self.prepared_model(*args, **kwargs)
        else:
            if self.quantized_model is None:
                self.quantized_model = convert_fx(self.prepared_model)
                print("quantized model:", self.quantized_model)
            print("running quantized model")
            return self.quantized_model(*args, **kwargs)

class QuantizationCompiler():
    def __init__(self, prepare_qconfig_dict, module_to_fqn):
        super().__init__()
        self.prepare_qconfig_dict = prepare_qconfig_dict
        self.module_to_fqn = module_to_fqn

    def enable_prepare(self):
        self.prepare_mode = True

    def enable_convert(self):
        self.prepare_mode = False

    def __call__(self, model, example_inputs):
        return QuantizationWrapper(self, model, example_inputs)

m = BasicModule().eval()
module_to_fqn = {mod: name for name, mod in m.named_modules()}
quant_compiler = QuantizationCompiler({"": default_qconfig}, module_to_fqn)

def quantize(m, example_inputs):
    torchdynamo.config.debug = True
    quant_compiler.enable_prepare()
    with torchdynamo.optimize(quant_compiler):
        # any PyTorch code
        # fx_prepare() is called to optimize extracted fragments
        # should reach a fixed point where nothing new is compiled
        m(*example_inputs)

    # calibration
    # Optionally:
    with torchdynamo.run():
        # any PyTorch code
        # previosly compiled artifacts are reused
        # provides a quiescence guarantee, without compiles
        m(*example_inputs)

    quant_compiler.enable_convert()
    with torchdynamo.run():
        # any PyTorch code
        # previosly compiled artifacts are reused
        # provides a quiescence guarantee, without compiles
        m(*example_inputs)

    print(m)

m = BasicModule().eval()
example_inputs = (torch.randn(1, 10),)    
quantize(m, example_inputs)

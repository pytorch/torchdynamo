import functools
import importlib
import itertools
import math
import operator
import os
import json
import logging
import torch

from torchdynamo.utils import torchscript, is_jit_model
from torchdynamo.testing import same
from torchdynamo import config

log = logging.getLogger(__name__)


def cached(fn):
    cached_name = f"_{fn.__name__}"

    @functools.wraps(fn)
    def inner(self):
        if hasattr(self, cached_name):
            return getattr(self, cached_name)
        result = fn(self)
        setattr(self, cached_name, result)
        return result

    return inner


def load_module_fx(name):
    pymod = importlib.import_module(f"subgraphs.{name}")
    # TODO(jansel): upstream these fixes to to_folder()
    pymod.module._operator_iadd = operator.iadd
    pymod.module._operator_imul = operator.imul
    pymod.module._operator_itruediv = operator.itruediv
    pymod.module._operator_setitem = operator.setitem
    pymod.module.math_sqrt = math.sqrt
    pymod.module.device = torch.device
    pymod.module.inf = float("inf")
    return pymod.FxModule()


def load_module_jit(name):
    filename = os.path.join(config.base_dir, "subgraphs", name, "model.ts")
    if not os.path.exists(filename):
        return None
    model = torch.jit.load(filename)
    assert is_jit_model(model)
    return model


class SubGraph(object):
    @classmethod
    def load(cls, name):
        model_dir = os.path.join(config.base_dir, "subgraphs", name)
        example_inputs = torch.load(os.path.join(model_dir, "example_inputs.pt"))
        example_outputs = torch.load(os.path.join(model_dir, "example_outputs.pt"))
        metadata = json.loads(open(os.path.join(model_dir, "metadata.json")).read())
        model_fx = load_module_fx(name)
        model_jit = load_module_jit(name)
        is_cuda = metadata["is_cuda"]
        if is_cuda:
            if model_fx is not None:
                model_fx = model_fx.cuda()
            if model_jit is not None:
                model_jit = model_jit.cuda()
            assert all(
                x.is_cuda for x in itertools.chain(example_inputs, example_outputs)
            )

        if model_jit is None:
            model_jit = torchscript(model_fx, example_inputs)
        if not same(example_outputs, model_fx(*example_inputs)):
            log.warning("FX graph is incorrect")
            assert model_jit and same(example_outputs, model_jit(*example_inputs))
            model_fx = model_jit

        subgraph = cls(model_fx, example_inputs, model_dir)
        subgraph._scripted = model_jit
        subgraph._example_outputs = example_outputs
        subgraph._is_cuda = is_cuda
        return subgraph

    def __init__(self, model, example_inputs, model_dir):
        super(SubGraph, self).__init__()
        self.model = model
        self.example_inputs = example_inputs
        self.model_dir = model_dir

    def filename(self, name):
        return os.path.join(self.model_dir, name)

    @property
    @cached
    def scripted(self):
        return torchscript(self.model, self.example_inputs)

    @property
    @cached
    def example_outputs(self):
        filename = self.filename("example_outputs.pt")
        if os.path.exists(filename):
            return torch.load(filename)
        result = self.model(*self.example_inputs)
        torch.save(result, filename)
        return result

    @property
    def example_outputs_list(self):
        if self.is_tensor_output:
            return [self.example_outputs]
        return self.example_outputs

    @property
    def input_names(self):
        return [f"i{i}" for i in range(len(self.example_inputs))]

    @property
    def is_tensor_output(self):
        return not isinstance(self.example_outputs, (list, tuple))

    @property
    def output_names(self):
        return [f"o{x}" for x in range(len(self.example_outputs_list))]

    @property
    def device_index(self):
        return 0

    @property
    @cached
    def onnx_filename(self):
        filename = self.filename("onnx")
        if os.path.exists(filename):
            return filename

        scripted = self.scripted
        example_inputs = self.example_inputs
        input_names = self.input_names
        output_names = self.output_names
        try:
            torch.onnx.export(
                scripted,
                example_inputs,
                filename,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=True,
                opset_version=14,
            )
        except IndexError:
            # work around bug in constant folding pass
            torch.onnx.export(
                scripted,
                example_inputs,
                filename,
                input_names=input_names,
                output_names=output_names,
                do_constant_folding=False,
                opset_version=14,
            )
        return filename

    @property
    def output_specs(self):
        return [
            (o.shape, o.dtype, o.layout, o.device, o.requires_grad)
            for o in self.example_outputs_list
        ]

    @property
    def is_cpu(self):
        return not self.is_cuda

    @property
    @cached
    def is_cuda(self):
        return self.example_inputs[0].device.type == "cuda"

    def empty_outputs_factory(self):
        specs = self.output_specs

        def create():
            return [
                torch.empty(
                    shape,
                    dtype=dtype,
                    layout=layout,
                    device=device,
                    requires_grad=requires_grad,
                )
                for shape, dtype, layout, device, requires_grad in specs
            ]

        return create

    def wrap_returns_list(self, fn):
        """Most backends return a list, but sometimes we want to return a tensor"""
        if self.is_tensor_output:
            return lambda *args: fn(*args)[0]
        return fn

    def wrap_returns_tensor(self, fn):
        """TRT backends auto-unpack len==1 lists, undo that"""
        if len(self.example_outputs_list) != 1 and self.is_tensor_output:
            return lambda *args: (fn(*args),)
        return fn

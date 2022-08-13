import json
import logging
import os
import zipfile
from collections import Counter
from collections import defaultdict
from functools import partial
from os.path import exists
from typing import Any
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils._pytree import tree_map

OP_INP_DIRECTORY = os.path.join(os.path.dirname(__file__), "operator_inp_logs")

TIMM_FILE = os.path.join(OP_INP_DIRECTORY, "timm_train_inps")
HF_FILE = os.path.join(OP_INP_DIRECTORY, "hf_train_inps")
TORCHBENCH_INPUT_PATH = os.path.join(OP_INP_DIRECTORY, "torchbench_train_inps")

aten = torch.ops.aten


def map_torch_args_to_json(e):
    if isinstance(e, torch.Tensor):
        fields = {
            "size": e.shape,
            "stride": e.stride(),
            "dtype": map_torch_args_to_json(e.dtype),
            "device": map_torch_args_to_json(e.device),
        }
        return {"torch.Tensor": fields}
    elif isinstance(e, torch.dtype):
        return {"torch.dtype": str(e)}
    elif isinstance(e, torch.device):
        return {"torch.device": f"torch.device('{e.type}')"}
    elif isinstance(e, torch.layout):
        return {"torch.layout": str(e)}
    else:
        return e


def map_json_to_torch_args(arg):
    if isinstance(arg, list):
        return [map_json_to_torch_args(elem) for elem in arg]
    elif isinstance(arg, tuple):
        return tuple(map_json_to_torch_args(elem) for elem in arg)
    elif isinstance(arg, dict):
        if "torch.Tensor" in arg:
            fields_dict = arg["torch.Tensor"]
            fields = ("size", "stride", "dtype", "device")
            kwargs = {
                field: map_json_to_torch_args(fields_dict[field]) for field in fields
            }
            return torch.empty_strided(**kwargs)
        for key in ("torch.dtype", "torch.device", "torch.layout"):
            if key in arg:
                return eval(arg[key])
        return {
            map_json_to_torch_args(k): map_json_to_torch_args(v) for k, v in arg.items()
        }
    else:
        return arg


def contains_tensor(elems):
    for elem in tree_flatten(elems)[0]:
        if isinstance(elem, torch.Tensor):
            return True
    return False


def skip_args(elems):
    for i in tree_flatten(elems)[0]:
        # only shows up in constructors and ops like that
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True
        # TODO: serialize/deserialize sparse arguments
        if isinstance(i, torch.Tensor) and i.is_sparse:
            return True
    return False


class OperatorInputsMode(TorchDispatchMode):
    def __init__(self, func_db=None):
        self.func_db = defaultdict(Counter) if func_db is None else func_db

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        arg_meta, kwarg_meta = tree_map(map_torch_args_to_json, (args, kwargs))

        out = func_overload(*args, **kwargs)

        inps = (args, kwargs)
        if contains_tensor(inps) and not skip_args(inps) and contains_tensor(out):
            json_str = json.dumps((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][json_str] += 1

        return out

    def log_to_file(self, output_filename):
        with open(output_filename, "w") as f:
            json.dump(self.func_db, f)


def map_to_device(e, device):
    return e.to(device) if isinstance(e, torch.Tensor) else e


def map_to_dtype(e, dtype):
    if isinstance(e, torch.Tensor) and e.is_floating_point():
        return e.to(dtype)
    else:
        return e


class OperatorInputsLoader:
    def __init__(self, json_file_path):
        with open(json_file_path, "r") as f:
            self.loaded_json_dict = json.load(f)

    def get_inputs_for_operator(
        self, operator, dtype, device="cuda"
    ) -> Generator[Tuple[Iterable[Any], Dict[str, Any]], None, None]:
        assert (
            str(operator) in self.loaded_json_dict
        ), f"Could not find {operator}, must provide overload"

        if "embedding" in str(operator):
            logging.warn("Embedding inputs NYI, input data cannot be randomized")
            yield
            return

        # counter represents number of times these inputs occured, ignored for now
        for inps, counter in self.loaded_json_dict[str(operator)].items():
            args, kwargs = json.loads(inps)
            args = map_json_to_torch_args(args)
            kwargs = map_json_to_torch_args(kwargs)

            to_dtype = partial(map_to_dtype, dtype=dtype)
            args, kwargs = tree_map(to_dtype, (args, kwargs))

            if device:
                to_device = partial(map_to_device, device=torch.device(device))
                args, kwargs = tree_map(to_device, (args, kwargs))

            yield args, kwargs

    def get_all_ops(self):
        for key in self.loaded_json_dict.keys():
            yield eval(key)

    def get_call_frequency(self, op):
        assert (
            str(op) in self.loaded_json_dict
        ), f"Could not find {op}, must provide overload"

        count = 0
        for _, counter in self.loaded_json_dict[str(op)].items():
            count += counter
        return count

    @staticmethod
    def get_timm_loader():
        return OperatorInputsLoader._get_loader(TIMM_FILE)

    @staticmethod
    def get_huggingface_loader():
        return OperatorInputsLoader._get_loader(HF_FILE)

    @staticmethod
    def get_torchbench_loader():
        return OperatorInputsLoader._get_loader(TORCHBENCH_INPUT_PATH)

    @staticmethod
    def _get_loader(inp_path):
        json_path = inp_path + ".json"
        if not exists(json_path):
            zip_path = inp_path + ".zip"
            assert exists(zip_path), f"Could not find {inp_path}"
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(OP_INP_DIRECTORY)

        return OperatorInputsLoader(json_path)

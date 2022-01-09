import collections
import dataclasses
import re
import types
from typing import Any

import torch

from . import skipfiles
from .allowed_functions import is_allowed
from .allowed_functions import is_builtin
from .guards import GuardBuilder
from .utils import getfile
from .utils import istensor
from .utils import istype
from .utils import warning
from .variable_source import GetItemSource
from .variable_source import Source
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import ListVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TupleVariable
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserDefinedClassVariable
from .variable_tracker import UserFunctionVariable
from .variable_tracker import typestr


@dataclasses.dataclass
class GraphArg:
    source: Source
    example: Any

    def load(self, tx):
        return self.source.reconstruct(tx)

    def get_examples(self):
        return [self.example]

    def __len__(self):
        return 1


class VariableBuilder:
    """Wrap a python value in a VariableTracker() instance"""

    def __init__(self, tx, source: Source):
        super(VariableBuilder, self).__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        return self._wrap(value).clone(**self.options())

    @staticmethod
    def list_type(value):
        return {
            tuple: TupleVariable,
            list: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
        }[type(value)]

    def get_source(self):
        return self.source

    def options(self):
        return {"source": self.get_source()}

    def make_guards(self, *guards):
        source = self.get_source()
        return {source.create_guard(guard) for guard in guards}

    def _wrap(self, value):
        make_guards = self.make_guards
        if istensor(value):
            return self.wrap_tensor(value)
        elif istype(value, (tuple, list)):
            guards = self.make_guards(GuardBuilder.LIST_LENGTH)
            output = [
                VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(
                    item
                ).add_guards(guards)
                for i, item in enumerate(value)
            ]
            return self.list_type(value)(output, guards=guards)
        elif istype(value, (dict, collections.OrderedDict)) and all(
            map(ConstantVariable.is_literal, value.keys())
        ):
            guards = self.make_guards(GuardBuilder.DICT_KEYS)
            keys = (
                value.keys()
                if istype(value, collections.OrderedDict)
                else sorted(value.keys())
            )
            result = collections.OrderedDict(
                (
                    k,
                    VariableBuilder(self.tx, GetItemSource(self.get_source(), k))(
                        value[k]
                    ).add_guards(guards),
                )
                for k in keys
            )
            return ConstDictVariable(result, guards=guards)
        elif isinstance(value, torch.nn.Module):
            return self.tx.add_submodule(
                value,
                self.name,
                source=self.get_source(),
                # Guards are added inside add_submodule
            )
        elif ConstantVariable.is_literal(value) or istype(value, torch.Size):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards=make_guards(GuardBuilder.CONSTANT_MATCH),
            )
        elif is_builtin(value):
            return BuiltinVariable(
                value,
                guards=make_guards(GuardBuilder.BUILTIN_MATCH),
            )
        elif is_allowed(value):
            return AllowedFunctionOrModuleVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, type) and not skipfiles.check(getfile(value)):
            return UserDefinedClassVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif istype(value, types.FunctionType) and not skipfiles.check(getfile(value)):
            return UserFunctionVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, types.ModuleType):
            return PythonModuleVariable(
                value,
                guards=make_guards(GuardBuilder.PYMODULE_MATCH),
            )

        else:
            warning(f"UnsupportedVariable {typestr(value)}")
            return self.wrap_unsupported(value)

    def wrap_unsupported(self, value):
        return UnsupportedVariable(
            value,
            guards=self.make_guards(GuardBuilder.TYPE_MATCH),
        )

    def wrap_tensor(self, value: torch.Tensor):
        if self.get_source().guard_source().is_nn_module():
            return self.tx.add_submodule(
                value,
                self.name,
                source=self.get_source(),
                # Gaurd sare done inside add_submodule
                # guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            )
        else:
            self.tx.graphargs.append(GraphArg(self.get_source(), value))
            return TensorVariable.create(
                proxy=self.tx.create_graph_input(
                    re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(value)
                ),
                example_value=value,
                guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            )

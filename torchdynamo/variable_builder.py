import dataclasses
import inspect
import types
from typing import List
from typing import Any

import torch

from . import skipfiles
from .allowed_functions import is_allowed
from .allowed_functions import is_builtin
from .guards import Guard
from .guards import GuardBuilder
from .guards import GuardSource
from .utils import istensor
from .utils import istype
from .utils import warning
from .variable_source import AttrSource
from .variable_source import GetItemSource
from .variable_source import GlobalSource
from .variable_source import LocalSource
from .variable_source import Source
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import NNModuleVariable
from .variable_tracker import BuiltinVariable
from .variable_tracker import ConstantVariable
from .variable_tracker import ConstDictVariable
from .variable_tracker import ListVariable
from .variable_tracker import PythonModuleVariable
from .variable_tracker import TensorVariable
from .variable_tracker import TupleVariable
from .variable_tracker import typestr
from .variable_tracker import UnsupportedVariable
from .variable_tracker import UserDefinedClassVariable
from .variable_tracker import UserFunctionVariable


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

    def __init__(self, tx, name: str):
        super(VariableBuilder, self).__init__()
        self.tx = tx
        self.name = name

    def __call__(self, value):
        return self._wrap(value).clone(**self.options())

    def _wrap(self, value):
        make_guards = self.make_guards
        if istensor(value):
            return self.wrap_tensor(value)
        elif (
            istype(value, (tuple, list, torch.nn.ParameterList))
            and value
            and all(istensor(x) for x in value)
        ):
            return self.wrap_tensor_list(value)
        # This would would pass floats into the graph as a input
        # elif istype(value, float):
        #     self.add_arg(name, value)
        #     return BasicTypeVariable(
        #         proxy=self.create_graph_input(name, type(value)),
        #         guards=make_guards(GuardBuilder.TYPE_MATCH),
        #     )
        elif (
            istype(value, (list, tuple, torch.nn.ModuleList))
            and value
            and all(isinstance(x, torch.nn.Module) for x in value)
        ):
            # TODO(jansel): add guards to check for mutation
            guards = self.make_guards(GuardBuilder.ID_MATCH)
            output = []
            for i, item in enumerate(value):
                output.append(
                    self.tx.add_submodule(
                        item,
                        self.name,
                        i,
                        source=GetItemSource(self.get_source(), i),
                        guards=guards,
                    )
                )
            return self.list_type(value)(output, guards=guards)
        elif isinstance(value, torch.nn.Module) and not istype(
            value, torch.nn.ModuleList
        ):
            # TODO(jansel): add a module guard type to check for training
            return self.tx.add_submodule(
                value,
                self.name,
                source=self.get_source(),
                guards=make_guards(
                    GuardBuilder.ID_MATCH,
                    # GuardBuilder.OBJECT_MUTATION,
                ),
            )
        elif value is None or istype(value, bool):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards=make_guards(GuardBuilder.ID_MATCH),
            )
        elif istype(value, (int, float, str)) or (
            istype(value, (tuple, list, torch.Size))
            and all(istype(x, int) for x in value)
        ):
            # For these, just specialize on exact value
            return ConstantVariable(
                value=value,
                guards=make_guards(GuardBuilder.EQUALS_MATCH),
            )
        elif is_builtin(value):
            return BuiltinVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif is_allowed(value):
            return AllowedFunctionOrModuleVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, type) and not skipfiles.check(inspect.getfile(value)):
            return UserDefinedClassVariable(
                value, guards=make_guards(GuardBuilder.ID_MATCH)
            )
        elif istype(value, types.FunctionType) and not skipfiles.check(
            inspect.getfile(value)
        ):
            return UserFunctionVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, types.ModuleType):
            return PythonModuleVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif (
            istype(value, dict)
            and value
            and all(map(istensor, value.values()))
            and all(map(ConstantVariable.is_literal, value.keys()))
        ):
            return self.wrap_tensor_dict(value)
        else:
            warning(f"UnsupportedVariable {typestr(value)}")
            return self.wrap_unsupported(value)

        assert False

    def list_type(self, value):
        return {
            tuple: TupleVariable,
            list: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
        }[type(value)]

    def wrap_tensor(self, value: torch.Tensor):
        self.add_arg(value)
        return TensorVariable(
            proxy=self.tx.create_graph_input(self.name, type(value)),
            guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            **TensorVariable.specialize(value),
        )

    def wrap_tensor_list(self, value: List[torch.Tensor]):
        guards = self.make_guards(GuardBuilder.FIXED_TENSOR_LIST)
        self.add_list_arg(value)
        items = [
            TensorVariable(
                proxy=self.tx.create_graph_input(f"{self.name}_{idx}", type(v)),
                guards=guards,
                **TensorVariable.specialize(v),
            )
            for idx, v in enumerate(value)
        ]
        return self.list_type(value)(items, guards=guards)

    def wrap_tensor_dict(self, value):
        guards = self.make_guards(GuardBuilder.FIXED_TENSOR_DICT)
        result = {}
        for k in sorted(value.keys()):
            v = value[k]
            self.add_getitem_arg(k, v)
            result[k] = TensorVariable(
                proxy=self.tx.create_graph_input(f"{self.name}_{k}", type(v)),
                guards=guards,
                **TensorVariable.specialize(v),
            )
        return ConstDictVariable(result, guards=guards)

    def wrap_unsupported(self, value):
        return UnsupportedVariable(
            value,
            guards=self.make_guards(GuardBuilder.TYPE_MATCH),
        )

    def add_arg(self, value):
        self.tx.graphargs.append(GraphArg(self.get_source(), value))

    def add_list_arg(self, value):
        for idx, item in enumerate(value):
            self.add_getitem_arg(idx, item)

    def add_getitem_arg(self, key, value):
        self.tx.graphargs.append(GraphArg(GetItemSource(self.get_source(), key), value))

    def make_guards(self, *guards):
        raise NotImplementedError()

    def get_source(self):
        raise NotImplementedError()

    def options(self):
        return {"source": self.get_source()}


class GlobalVariableBuilder(VariableBuilder):
    def make_guards(self, *guards):
        return {
            Guard(self.name, GuardSource.GLOBAL, guard)
            for guard in guards
            # Skip guards on global functions
            if guard is not GuardBuilder.FUNCTION_MATCH
        }

    def get_source(self):
        return GlobalSource(self.name)


class LocalVariableBuilder(VariableBuilder):
    def make_guards(self, *guards):
        return {Guard(self.name, GuardSource.LOCAL, guard) for guard in guards}

    def get_source(self):
        return LocalSource(self.name)


class AttributeVariableBuilder(VariableBuilder):
    def __init__(self, tx, base_var, attr_name, guards):
        assert istype(base_var, NNModuleVariable)
        self.base_var = base_var
        self.attr_name = attr_name
        self._guards = guards or set()
        super().__init__(tx, self.get_source().name())

    def make_guards(self, *guards):
        return {
            self.get_source().create_guard(guard) for guard in guards
        } | self._guards

    def get_source(self):
        return AttrSource(self.base_var.source, self.attr_name)

    def wrap_tensor(self, value: torch.Tensor):
        return self.tx.add_submodule(
            value,
            self.name,
            # guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            source=self.get_source(),
            **TensorVariable.specialize(value),
        )

    def wrap_tensor_list(self, value: List[torch.Tensor]):
        output = []
        # guards = self.make_guards(GuardBuilder.FIXED_TENSOR_LIST)
        guards = set()
        for i, item in enumerate(value):
            output.append(
                self.tx.add_submodule(
                    item,
                    self.name,
                    i,
                    guards=guards,
                    **TensorVariable.specialize(item),
                )
            )
        return self.list_type(value)(output, guards=guards)

    def wrap_tensor_dict(self, value):
        return self.wrap_unsupported(value)

    # def wrap_unsupported(self, value):
    #    return GetAttrVariable(
    #        self.base_var,
    #        self.attr_name,
    #        guards=self.make_guards(),
    #    )

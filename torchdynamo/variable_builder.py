import dataclasses
import inspect
import types
from typing import Any

import torch

from . import skipfiles
from .allowed_functions import is_allowed
from .allowed_functions import is_builtin
from .bytecode_transformation import create_instruction
from .guards import Guard
from .guards import GuardBuilder
from .guards import GuardSource
from .utils import istensor
from .utils import istype
from .utils import warning
from .variable_source import LocalSource, GlobalSource
from .variable_tracker import AllowedFunctionOrModuleVariable
from .variable_tracker import BuiltinVariable
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
class Arg:
    name: str
    example: Any

    def get_examples(self):
        return [self.example]

    def __len__(self):
        return 1


@dataclasses.dataclass
class LocalArg(Arg):
    def load(self, tracer):
        return [tracer.create_load(self.name)]


@dataclasses.dataclass
class GlobalArg(Arg):
    def load(self, tracer):
        return [tracer.create_load_global(self.name)]


@dataclasses.dataclass
class ListArg(Arg):
    length: int

    def get_examples(self):
        return list(reversed(self.example))

    def __len__(self):
        return self.length


@dataclasses.dataclass
class LocalListArg(ListArg):
    def load(self, tracer):
        return [
            tracer.create_load(self.name),
            create_instruction("UNPACK_SEQUENCE", self.length),
        ]


@dataclasses.dataclass
class GlobalListArg(ListArg):
    def load(self, tracer):
        return [
            tracer.create_load_global(self.name),
            create_instruction("UNPACK_SEQUENCE", self.length),
        ]


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
            self.add_arg(value)
            return TensorVariable(
                proxy=self.tx.create_graph_input(self.name, type(value)),
                guards=make_guards(GuardBuilder.TENSOR_MATCH),
                **TensorVariable.specialize(value),
            )
        elif istype(value, (tuple, list)) and value and all(istensor(x) for x in value):
            self.add_list_arg(value)
            items = [
                TensorVariable(
                    proxy=self.tx.create_graph_input(f"{self.name}_{idx}", type(v)),
                    guards=make_guards(GuardBuilder.FIXED_TENSOR_LIST),
                    **TensorVariable.specialize(v),
                )
                for idx, v in reversed(list(enumerate(value)))
            ]
            cls = {tuple: TupleVariable, list: ListVariable}[type(value)]
            return cls(
                list(reversed(items)),
                guards=make_guards(GuardBuilder.FIXED_TENSOR_LIST),
            )
        # This would would pass floats into the graph as a input
        # elif istype(value, float):
        #     self.add_arg(name, value)
        #     return BasicTypeVariable(
        #         proxy=self.create_graph_input(name, type(value)),
        #         guards=make_guards(GuardBuilder.TYPE_MATCH),
        #     )
        elif isinstance(value, torch.nn.Module):
            return self.tx.add_submodule(
                value,
                self.name,
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
        elif istype(value, (int, float)) or (
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
        else:
            warning(f"UnsupportedVariable {typestr(value)}")
            return UnsupportedVariable(
                value,
                guards=make_guards(GuardBuilder.TYPE_MATCH),
            )

    def add_arg(self, value):
        raise NotImplementedError()

    def add_list_arg(self, value):
        raise NotImplementedError()

    def make_guards(self, *guards):
        raise NotImplementedError()

    def options(self):
        raise NotImplementedError()


class GlobalVariableBuilder(VariableBuilder):
    def add_arg(self, value):
        self.tx.graphargs.append(GlobalArg(self.name, value))

    def add_list_arg(self, value):
        self.tx.graphargs.append(GlobalListArg(self.name, value, len(value)))

    def make_guards(self, *guards):
        return {
            Guard(self.name, GuardSource.GLOBAL, guard)
            for guard in guards
            # Skip guards on global functions
            if guard is not GuardBuilder.FUNCTION_MATCH
        }

    def options(self):
        return {"source": GlobalSource(self.name)}


class LocalVariableBuilder(VariableBuilder):
    def add_arg(self, value):
        self.tx.graphargs.append(LocalArg(self.name, value))

    def add_list_arg(self, value):
        self.tx.graphargs.append(LocalListArg(self.name, value, len(value)))

    def make_guards(self, *guards):
        return {Guard(self.name, GuardSource.LOCAL, guard) for guard in guards}

    def options(self):
        return {"source": LocalSource(self.name)}

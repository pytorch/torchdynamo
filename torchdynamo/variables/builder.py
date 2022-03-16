import collections
import dataclasses
import functools
import inspect
import re
import types
from typing import Any

import numpy as np
import torch

import torchdynamo

from .. import mutation_guard
from .. import skipfiles
from ..allowed_functions import is_allowed
from ..allowed_functions import is_builtin
from ..allowed_functions import is_numpy
from ..exc import unimplemented
from ..guards import GuardBuilder
from ..side_effects import SideEffects
from ..source import AttrSource
from ..source import GetItemSource
from ..source import Source
from ..source import TupleIteratorGetItemSource
from ..utils import getfile
from ..utils import is_namedtuple
from ..utils import istensor
from ..utils import istype
from ..utils import tuple_iterator
from ..utils import tuple_iterator_getitem
from ..utils import tuple_iterator_len
from .base import MutableLocal
from .builtin import BuiltinVariable
from .constant import ConstantVariable
from .dicts import ConstDictVariable
from .dicts import DataClassVariable
from .functions import UserFunctionVariable
from .lists import ListIteratorVariable
from .lists import ListVariable
from .lists import NamedTupleVariable
from .lists import RangeVariable
from .lists import TupleVariable
from .misc import AutogradFunctionVariable
from .misc import InspectSignatureVariable
from .misc import LambdaVariable
from .misc import NumpyVariable
from .misc import PythonModuleVariable
from .misc import SkipFilesVariable
from .nn_module import UnspecializedNNModuleVariable
from .tensor import TensorVariable
from .torch import TorchVariable
from .user_defined import UserDefinedClassVariable
from .user_defined import UserDefinedObjectVariable


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

    def __init__(
        self,
        tx: "torchdynamo.symbolic_convert.InstructionTranslatorBase",
        source: Source,
    ):
        super(VariableBuilder, self).__init__()
        self.tx = tx
        self.source = source
        self.name = source.name()

    def __call__(self, value):
        if value in self.tx.output.side_effects:
            # TODO(jansel): add guard for alias relationship
            return self.tx.output.side_effects[value]
        return self._wrap(value).clone(**self.options())

    @staticmethod
    def list_type(value):
        if is_namedtuple(value):
            return functools.partial(NamedTupleVariable, tuple_cls=type(value))
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
        elif istype(value, (tuple, list)) or is_namedtuple(value):
            guards = self.make_guards(GuardBuilder.LIST_LENGTH)
            output = [
                VariableBuilder(self.tx, GetItemSource(self.get_source(), i))(
                    item
                ).add_guards(guards)
                for i, item in enumerate(value)
            ]
            result = self.list_type(value)(output, guards=guards)
            if istype(value, list):
                return self.tx.output.side_effects.track_list(
                    self.source, value, result
                )
            return result
        elif istype(value, tuple_iterator):
            guards = self.make_guards(GuardBuilder.TUPLE_ITERATOR_LEN)
            output = [
                VariableBuilder(
                    self.tx, TupleIteratorGetItemSource(self.get_source(), i)
                )(tuple_iterator_getitem(value, i)).add_guards(guards)
                for i in range(tuple_iterator_len(value))
            ]
            return ListIteratorVariable(
                output, mutable_local=MutableLocal(), guards=guards
            )
        elif istype(value, range):
            guards = self.make_guards(GuardBuilder.EQUALS_MATCH)
            return RangeVariable(value=value, guards=guards)
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
            result = ConstDictVariable(result, guards=guards)
            if istype(value, dict):
                return self.tx.output.side_effects.track_dict(
                    self.source, value, result
                )
            return result
        elif isinstance(value, torch.nn.Module):
            if mutation_guard.is_dynamic_nn_module(value):
                # created dynamically, don't specialize on it
                result = UnspecializedNNModuleVariable(
                    value, guards=make_guards(GuardBuilder.TYPE_MATCH)
                )
                if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                    # don't allow STORE_ATTR mutation with custom __setattr__
                    return result
                return self.tx.output.side_effects.track_object_existing(
                    self.source, value, result
                )
            else:
                return self.tx.output.add_submodule(
                    value,
                    self.name,
                    source=self.get_source(),
                    # Guards are added inside add_submodule
                )
        elif ConstantVariable.is_literal(value) or istype(
            value, (torch.Size, torch.device, torch.dtype)
        ):
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
            return TorchVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif value is inspect.signature:
            return LambdaVariable(
                InspectSignatureVariable.create,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif value is dataclasses.fields:
            return LambdaVariable(
                _dataclasses_fields_lambda,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif is_numpy(value):
            return NumpyVariable(
                value,
                guards=make_guards(
                    GuardBuilder.FUNCTION_MATCH
                    if callable(value)
                    else GuardBuilder.TYPE_MATCH
                ),
            )
        elif (
            istype(value, (type, types.FunctionType))
            and skipfiles.check(getfile(value), allow_torch=True)
            and not inspect.getattr_static(value, "_torchdynamo_inline", False)
        ):
            return SkipFilesVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif istype(value, type):
            return UserDefinedClassVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        elif istype(value, types.FunctionType):
            return UserFunctionVariable(
                value,
                guards=make_guards(GuardBuilder.FUNCTION_MATCH),
            )
        elif istype(value, types.ModuleType):
            return PythonModuleVariable(
                value,
                guards=make_guards(GuardBuilder.PYMODULE_MATCH),
            )
        elif type(value) is torch.autograd.function.FunctionMeta:
            return AutogradFunctionVariable(
                value, guards=make_guards(GuardBuilder.FUNCTION_MATCH)
            )
        if istype(
            value,
            (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return self._wrap(int(value))
        elif DataClassVariable.is_matching_object(value):
            return DataClassVariable.wrap(self, value).add_guards(
                make_guards(GuardBuilder.TYPE_MATCH)
            )
        else:
            result = UserDefinedObjectVariable(
                value,
                guards=self.make_guards(GuardBuilder.TYPE_MATCH),
            )
            if not SideEffects.cls_supports_mutation_side_effects(type(value)):
                # don't allow STORE_ATTR mutation with custom __setattr__
                return result
            return self.tx.output.side_effects.track_object_existing(
                self.source, value, result
            )

    def wrap_tensor(self, value: torch.Tensor):
        if self.get_source().guard_source().is_nn_module():
            return self.tx.output.add_submodule(
                value,
                self.name,
                source=self.get_source(),
                # Guards are done inside add_submodule
                # guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            )
        else:
            self.tx.output.graphargs.append(GraphArg(self.get_source(), value))
            return TensorVariable.create(
                tx=self.tx,
                proxy=self.tx.output.create_graph_input(
                    re.sub(r"[^a-zA-Z0-9]+", "_", self.name), type(value)
                ),
                example_value=value,
                guards=self.make_guards(GuardBuilder.TENSOR_MATCH),
            )


def _dataclasses_fields_lambda(obj):
    if isinstance(obj, UserDefinedObjectVariable):
        value = obj.value
    elif isinstance(obj, DataClassVariable):
        value = obj.user_cls
    else:
        unimplemented(f"Dataclass fields handling fails for type {obj}")
    items = []
    for field in dataclasses.fields(value):
        source = None
        if obj.source:
            source = GetItemSource(
                AttrSource(obj.source, "__dataclass_fields__"), field.name
            )
        items.append(UserDefinedObjectVariable(field, source=source).add_options(obj))
    return TupleVariable(items).add_options(obj)

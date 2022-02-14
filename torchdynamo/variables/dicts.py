import collections
from typing import Dict
from typing import List

from ..bytecode_transformation import create_instruction
from .base import VariableTracker


class ConstDictVariable(VariableTracker):
    def __init__(self, items, **kwargs):
        super(ConstDictVariable, self).__init__(**kwargs)
        if not isinstance(items, collections.OrderedDict):
            assert isinstance(items, dict)
            items = collections.OrderedDict((k, items[k]) for k in sorted(items.keys()))
        self.items = items

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def python_type(self):
        return dict

    def reconstruct(self, codegen):
        if len(self.items) == 0:
            return [create_instruction("BUILD_MAP", 0)]
        keys = tuple(sorted(self.items.keys()))
        for key in keys:
            codegen(self.items[key])
        return [
            codegen.create_load_const(keys),
            create_instruction("BUILD_CONST_KEY_MAP", len(keys)),
        ]

    def getitem_const(self, arg: VariableTracker):
        index = arg.as_python_constant()
        return self.items[index].add_options(self, arg)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from . import ConstantVariable
        from . import TupleVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        val = self.items

        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            return self.getitem_const(args[0])
        elif name == "items":
            assert not (args or kwargs)
            return TupleVariable(
                [
                    TupleVariable([ConstantVariable(k, **options), v], **options)
                    for k, v in val.items()
                ],
                **options,
            )
        elif name == "keys":
            assert not (args or kwargs)
            return TupleVariable(
                [ConstantVariable(k, **options) for k in val.keys()],
                **options,
            )

        elif name == "values":
            assert not (args or kwargs)
            return TupleVariable(list(val.values()), **options)
        elif name == "__len__":
            assert not (args or kwargs)
            return ConstantVariable(len(self.items), **options)
        elif (
            name == "__setitem__"
            and args
            and args[0].is_python_constant()
            and self.mutable_local
        ):
            assert not kwargs and len(args) == 2
            newval = collections.OrderedDict(val)
            newval[args[0].as_python_constant()] = args[1]
            return tx.replace_all(self, ConstDictVariable(newval, **options))
        elif (
            name in ("pop", "get")
            and args
            and args[0].is_python_constant()
            and args[0].as_python_constant() not in self.items
            and len(args) == 2
        ):
            # missing item, return the default value
            return args[1].add_options(options)
        elif (
            name == "pop"
            and args
            and args[0].is_python_constant()
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            result = newval.pop(args[0].as_python_constant())
            tx.replace_all(self, ConstDictVariable(newval, **options))
            return result.add_options(options)
        elif (
            name == "update"
            and args
            and isinstance(args[0], ConstDictVariable)
            and self.mutable_local
        ):
            newval = collections.OrderedDict(val)
            newval.update(args[0].items)
            result = ConstDictVariable(newval, **options)
            return tx.replace_all(self, result)
        elif (
            name in ("get", "__getattr__")
            and args
            and args[0].is_python_constant()
            and args[0].as_python_constant() in self.items
        ):
            result = self.items[args[0].as_python_constant()]
            return result.add_options(options)
        elif name == "__contains__" and args and args[0].is_python_constant():
            return ConstantVariable(
                args[0].as_python_constant() in self.items, **options
            )
        else:
            return super().call_method(tx, name, args, kwargs)

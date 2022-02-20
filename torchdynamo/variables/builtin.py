import functools
import itertools
import math
import operator
from typing import Dict
from typing import List

import torch

from .. import variables
from ..bytecode_transformation import create_instruction
from ..utils import check_constant_args
from ..utils import istype
from ..utils import proxy_args_kwargs
from ..utils import unimplemented
from .base import MutableLocal
from .base import VariableTracker


class BuiltinVariable(VariableTracker):
    @staticmethod
    @functools.lru_cache(None)
    def _constant_fold_functions():
        fns = {
            abs,
            all,
            any,
            bool,
            callable,
            chr,
            dict,
            divmod,
            float,
            int,
            len,
            list,
            max,
            min,
            ord,
            pow,
            repr,
            round,
            set,
            str,
            sum,
            tuple,
            type,
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        fns.update(x for x in math.__dict__.values() if isinstance(x, type(math.sqrt)))
        return fns

    def can_constant_fold_through(self):
        return self.fn in self._constant_fold_functions()

    @staticmethod
    @functools.lru_cache(None)
    def _fx_graph_functions():
        fns = {
            operator.pos,
            operator.neg,
            operator.not_,
            operator.invert,
            operator.pow,
            operator.mul,
            operator.matmul,
            operator.floordiv,
            operator.truediv,
            operator.mod,
            operator.add,
            operator.sub,
            operator.getitem,
            operator.lshift,
            operator.rshift,
            operator.and_,
            operator.or_,
            operator.xor,
            operator.ipow,
            operator.imul,
            operator.imatmul,
            operator.ifloordiv,
            operator.itruediv,
            operator.imod,
            operator.iadd,
            operator.isub,
            operator.ilshift,
            operator.irshift,
            operator.iand,
            operator.ixor,
            operator.ior,
        }
        return fns

    def can_insert_in_graph(self):
        return self.fn in self._fx_graph_functions()

    def __init__(self, fn, **kwargs):
        super(BuiltinVariable, self).__init__(**kwargs)
        self.fn = fn

    def __str__(self):
        return f"{self.__class__.__name__}({self.fn.__name__})"

    def python_type(self):
        return type(self.fn)

    def as_python_constant(self):
        return self.fn

    def reconstruct(self, codegen):
        name = self.fn.__name__
        assert self.fn.__module__ == "builtins"
        assert name not in codegen.tx.f_globals, "shadowed global"
        return [codegen.create_load_global(name, add=True)]

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        constant_args = check_constant_args(args, kwargs)
        tensor_args = any(
            isinstance(i, variables.TensorVariable)
            for i in itertools.chain(args, kwargs.values())
        )
        options = VariableTracker.propagate(self, args, kwargs.values())
        assert isinstance(args, list)
        assert isinstance(kwargs, dict)

        if self.can_insert_in_graph() and tensor_args:
            try:
                fn = self.fn
                if self.fn is operator.iadd and isinstance(
                    args[0], variables.ConstantVariable
                ):
                    # Work around weird bug in hf_T5
                    fn, args = operator.add, [args[1], args[0]]
                return variables.TensorVariable.create(
                    tx,
                    tx.output.create_proxy(
                        "call_function",
                        fn,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                    **options,
                )
            except NotImplementedError:
                unimplemented(f"partial tensor op: {self} {args} {kwargs}")
        elif self.fn in (min, max) and tensor_args:
            a, b = args
            if not isinstance(a, variables.TensorVariable):
                a, b = b, a
            assert isinstance(a, variables.TensorVariable) and not kwargs
            # convert min/max to torch ops
            if b.is_python_constant():
                kwargs = {"min": b} if (self.fn is max) else {"max": b}
                return variables.TorchVariable(torch.clamp, **options).call_function(
                    tx, [a], kwargs
                )
            else:
                fn = {max: torch.maximum, min: torch.minimum}[self.fn]
                return variables.TorchVariable(fn, **options).call_function(
                    tx, [a, b], {}
                )
        elif self.fn is range and constant_args:
            return variables.RangeVariable(
                value=range(
                    *[x.value for x in args],
                    **{k: v.value for k, v in kwargs.items()},
                ),
                **options,
            )
        elif self.fn is slice:
            assert not kwargs
            return variables.SliceVariable(args, **options)
        elif (
            self.fn is iter and args and isinstance(args[0], variables.BaseListVariable)
        ):
            assert not kwargs and len(args) == 1
            return variables.ListIteratorVariable(
                args[0].items, mutable_local=MutableLocal(), **options
            )
        elif (
            self.fn in (iter, tuple, list)
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs and len(args) == 1
            cls = variables.BaseListVariable.cls_for(self.fn)
            return cls(
                list(args[0].unpack_var_sequence(tx)),
                mutable_local=MutableLocal(),
                **options,
            )
        elif self.fn is zip and all(x.has_unpack_var_sequence(tx) for x in args):
            assert not kwargs
            items = [
                variables.TupleVariable(list(item), **options)
                for item in zip(*[arg.unpack_var_sequence(tx) for arg in args])
            ]
            return variables.TupleVariable(items, **options)
        elif self.fn is zip and all(
            isinstance(x, variables.TensorVariable) and x.size for x in args
        ):
            out_size = functools.reduce(min, [x.size[0] for x in args])
            items = []
            for i in range(out_size):
                items.append(
                    variables.TupleVariable(
                        [
                            BuiltinVariable(operator.getitem, **options).call_function(
                                tx, [arg, variables.ConstantVariable(i)], {}
                            )
                            for arg in args
                        ],
                        **options,
                    )
                )
            return variables.TupleVariable(items, **options)
        elif self.fn is enumerate and all(x.has_unpack_var_sequence(tx) for x in args):
            assert not kwargs and len(args) == 1
            items = [
                variables.TupleVariable(
                    [variables.ConstantVariable(idx, **options), var],
                    **options,
                )
                for idx, var in enumerate(args[0].unpack_var_sequence(tx))
            ]
            return variables.TupleVariable(items, **options)
        elif (
            self.fn is operator.mul
            and isinstance(
                args[0],
                (
                    variables.ListVariable,
                    variables.TupleVariable,
                ),
            )
            and isinstance(args[1], variables.ConstantVariable)
        ):
            assert not kwargs and len(args) == 2
            return args[0].__class__(
                args[0].items * args[1].as_python_constant(), **options
            )
        elif (
            self.fn is operator.mul
            and isinstance(
                args[1],
                (
                    variables.ListVariable,
                    variables.TupleVariable,
                ),
            )
            and isinstance(args[0], variables.ConstantVariable)
        ):
            return self.call_function(tx, list(reversed(args)), {})
        elif self.can_constant_fold_through() and constant_args:
            # constant fold
            return variables.ConstantVariable(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
                **options,
            )
        elif self.fn is len:
            return args[0].call_method(tx, "__len__", args[1:], kwargs)
        elif self.fn is operator.add:
            return args[0].call_method(tx, "__add__", args[1:], kwargs)
        elif self.fn is operator.getitem:
            return args[0].call_method(tx, "__getitem__", args[1:], kwargs)
        elif self.fn is isinstance:
            assert not kwargs and len(args) == 2
            arg, isinstance_type = args
            arg_type = arg.python_type()
            isinstance_type = isinstance_type.as_python_constant()
            try:
                val = issubclass(arg_type, isinstance_type)
            except TypeError:
                val = arg_type is isinstance_type
            return variables.ConstantVariable(val, **options)
        elif self.fn is super:
            assert not kwargs
            assert len(args) in (1, 2)
            return variables.SuperVariable(*args, **options)
        elif (
            self.fn is next
            and args
            and isinstance(args[0], variables.ListIteratorVariable)
        ):
            val, next_iter = args[0].next_variables()
            tx.replace_all(args[0], next_iter)
            return val.add_options(self)
        elif self.fn is hasattr and args:
            assert not kwargs
            obj, attr = args
            name = attr.as_python_constant()
            return obj.call_hasattr(tx, name).add_options(options)
        elif self.fn is map and args[1].has_unpack_var_sequence(tx):
            assert not kwargs and len(args) == 2
            items = [
                args[0].call_function(tx, [x], {})
                for x in args[1].unpack_var_sequence(tx)
            ]
            return variables.TupleVariable(items, **options)
        elif self.fn is sum and args[0].has_unpack_var_sequence(tx):
            start = kwargs.pop(
                "start", variables.ConstantVariable(0)
            ).as_python_constant()
            assert len(args) == 1 and not kwargs
            items = args[0].unpack_var_sequence(tx)[start:]
            return BuiltinVariable(functools.reduce, **options).call_function(
                tx,
                [
                    BuiltinVariable(operator.add),
                    variables.TupleVariable(items),
                    variables.ConstantVariable(0, **options),
                ],
                {},
            )
        elif self.fn is functools.reduce and args[1].has_unpack_var_sequence(tx):
            assert not kwargs or len(args) in (2, 3)
            items = args[1].unpack_var_sequence(tx)
            if len(args) == 2:
                value, items = items[0], items[1:]
            else:
                value = args[2]
            for element in items:
                value = args[0].call_function(tx, [value, element], {})
            return value.add_options(options)
        elif (
            self.fn is getattr
            and len(args) == 2
            and isinstance(args[1], variables.ConstantVariable)
            and istype(args[1].value, str)
        ):
            tx.push(args[0])
            tx.LOAD_ATTR(create_instruction("LOAD_ATTR", args[1].value))
            val = tx.pop()
            return val.add_options(options)
        else:
            raise super().call_function(tx, args, kwargs)

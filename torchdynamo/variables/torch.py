import re
import types
from typing import Dict
from typing import List

import torch._C
import torch.nn

from .. import config
from .. import variables
from ..allowed_functions import _allowed_function_ids
from ..exc import unimplemented
from ..utils import check_constant_args
from ..utils import istype
from ..utils import product
from ..utils import proxy_args_kwargs
from .base import VariableTracker


class TorchVariable(VariableTracker):
    """Points to a module or method in torch.*"""

    def __init__(self, value, **kwargs):
        super(TorchVariable, self).__init__(**kwargs)

        self.value = value

        # the remainder of this is just optional debug checks
        try:
            self_should_be_none = getattr(self.value, "__self__", None)
        except RuntimeError as e:
            assert "No such operator" in str(e), str(e)
            self_should_be_none = None

        # assert "_ntuple.<locals>.parse" not in str(value)

        if self_should_be_none is None:
            pass
        elif isinstance(self_should_be_none, types.ModuleType):
            # weird ones like torch.nn.functional.avg_pool2d have __self__
            name = self_should_be_none.__name__
            assert re.match(r"^(torch|math)([.]|$)", name), f"__self__ set to {name}"
        elif isinstance(
            self_should_be_none, type(torch._C._get_tracing_state.__self__)
        ):
            # some _C functions have __self__ as a null capsule
            pass
        else:
            assert False, f"{value} found with __self__ set"

    def unique_var_name(self):
        name = _allowed_function_ids().get(
            id(self.value), f"allowed_fn_{id(self.value)}"
        )
        return "__" + re.sub(r"[^a-zA-Z0-9_]+", "_", name)

    def reconstruct(self, codegen):
        return codegen.setup_globally_cached(self.unique_var_name(), self.value)

    def as_proxy(self):
        return self.value

    def python_type(self):
        if isinstance(self.value, (torch.Tensor, torch.nn.Module)):
            return type(self.value)
        return super().python_type()

    def as_python_constant(self):
        return self.value

    def can_constant_fold_through(self):
        if self.value in (torch.is_tensor, torch.is_floating_point):
            return True
        return getattr(self.value, "__module__", None) == "math"

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from . import ConstantVariable
        from . import GradModeVariable
        from . import TensorVariable

        constant_args = check_constant_args(args, kwargs)
        options = VariableTracker.propagate(self, args, kwargs.values())

        if self.value in config.constant_functions:
            assert not args and not kwargs
            return ConstantVariable(config.constant_functions[self.value], **options)
        elif self.can_constant_fold_through() and constant_args:
            # constant fold
            return ConstantVariable(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
                **options,
            )
        elif istype(self.value, type) and issubclass(self.value, torch.nn.Module):
            if self.value is torch.nn.Softmax:
                return self._call_softmax(tx, args, kwargs, options)
            else:
                unimplemented(f"construct nn.Module: {self.value.__name__}")
        elif (
            self.value in (torch.is_tensor, torch.is_floating_point)
            and isinstance(args[0], TensorVariable)
            and args[0].dtype is not None
        ):
            if self.value is torch.is_tensor:
                return ConstantVariable(True, **options)
            elif self.value is torch.is_floating_point:
                return ConstantVariable(args[0].dtype.is_floating_point, **options)
            else:
                assert False
        elif (
            self.value is torch.numel
            and isinstance(args[0], TensorVariable)
            and args[0].size is not None
        ):
            return ConstantVariable(product(args[0].size), **options)
        elif self.value in (
            torch.nn.modules.utils._single,
            torch.nn.modules.utils._pair,
            torch.nn.modules.utils._triple,
            torch.nn.modules.utils._quadruple,
            torch.nn.modules.utils._ntuple,
        ):
            return self._call_ntuple(tx, args, kwargs, options)
        elif self.value is torch.no_grad:
            return GradModeVariable(False, **options)
        elif self.value is torch.enable_grad:
            return GradModeVariable(True, **options)
        elif self.value is torch.set_grad_enabled and len(args) == 1:
            return GradModeVariable(args[0].as_python_constant(), **options)
        elif self.value is torch.is_grad_enabled:
            assert not (args or kwargs)
            return ConstantVariable(torch.is_grad_enabled(), **options).add_guards(
                GradModeVariable._guards_singleton
            )
        elif not config.dynamic_shapes and self.is_dynamic_shapes(args, kwargs):
            unimplemented(f"dynamic shapes: {self.value.__name__}")
        else:
            return TensorVariable.create(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function", self.value, *proxy_args_kwargs(args, kwargs)
                ),
                **options,
            )

    def is_dynamic_shapes(self, args, kwargs):
        """Check for dynamic shapes when shape specialization is enabled"""
        # TODO(jansel): need to get a complete list
        if self.value in (
            torch.nonzero,
            torch.unique,
            torch.unique_consecutive,
        ) or self.value.__name__ in ("nms",):
            return True

        if self.value is torch.where and len(args) + len(kwargs) == 1:
            return True

        if self.value in (
            torch.arange,
            torch.repeat_interleave,
        ):
            none = variables.ConstantVariable(None)

            def has_non_const(it):
                return not all(x.is_python_constant() for x in it)

            def arange(start=none, end=none, step=none, **kwargs):
                return has_non_const([start, end, step])

            def repeat_interleave(input, repeats, dim=none, **kwargs):
                return has_non_const([repeats])

            return locals()[self.value.__name__](*args, **kwargs)

        return False

    def _call_softmax(self, tx, args, kwargs, options):
        """rewrite the pattern nn.Softmax(dim=-1)(x) to F.softmax(x, -1)"""
        dim = args[0] if args else kwargs.get("dim", variables.ConstantVariable(None))

        def fake_softmax(input):
            return variables.TensorVariable.create(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.nn.functional.softmax,
                    *proxy_args_kwargs([input, dim], {}),
                ),
                **VariableTracker.propagate([self, dim, input]),
            )

        return variables.LambdaVariable(fake_softmax, **options)

    def _call_ntuple(self, tx, args, kwargs, options):
        """inline behavior of torch.nn.modules.utils._ntuple"""
        if self.value is torch.nn.modules.utils._ntuple:
            count = args[0].as_python_constant()
        else:
            count = self.value.__closure__[0].cell_contents
        assert isinstance(count, int)

        def handle_ntuple(value):
            if value.has_unpack_var_sequence(tx):
                return variables.TupleVariable(
                    list(value.unpack_var_sequence(tx)),
                    **VariableTracker.propagate(self, value, args, kwargs.values()),
                )
            elif value.is_python_constant():
                # constant prop through it
                return variables.ConstantVariable(
                    torch.nn.modules.utils._ntuple(count)(value.as_python_constant()),
                    **VariableTracker.propagate(self, value, args, kwargs.values()),
                )
            else:
                unimplemented(f"torch.nn.modules.utils._ntuple({value})")

        if self.value is torch.nn.modules.utils._ntuple:
            return variables.LambdaVariable(handle_ntuple, **options)
        else:
            return handle_ntuple(args[0])

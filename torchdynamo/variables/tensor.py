import copy
import itertools
import operator
from typing import Dict
from typing import List

import torch.fx
import torch.random

from .. import config
from .. import variables
from ..exc import unimplemented
from ..utils import clone_tensor
from ..utils import istype
from ..utils import product
from ..utils import proxy_args_kwargs
from .base import VariableTracker
from .base import typestr
from .lists import SizeVariable


class TensorVariable(VariableTracker):
    """A torch.Tensor input or an intermediate value in the FX graph"""

    _nonvar_fields = [
        "proxy",
        "dtype",
        "device",
        "ndim",
        "size",
        "stride",
        "requires_grad",
        "is_quantized",
    ]

    @staticmethod
    def propagate_args_kwargs(node):
        def visit(n: torch.fx.Node):
            return n.meta["example_value"]

        return torch.fx.node.map_arg((node.args, node.kwargs), visit)

    @classmethod
    def create(cls, tx, proxy, example_value=None, nnmodule=None, **options):
        if "guards" in options:
            tx.output.guards.update(options["guards"])
        assert "example_value" not in proxy.node.meta
        if not config.dynamic_propagation:
            if isinstance(example_value, torch.Tensor):
                options.update(TensorVariable.specialize(example_value))
            return TensorVariable(proxy, **options)

        if example_value is None:
            rng = torch.clone(torch.random.get_rng_state())
            if torch.cuda.is_available():
                cuda_rng = torch.clone(torch.cuda.get_rng_state())
            op = proxy.node.op
            args, kwargs = cls.propagate_args_kwargs(proxy.node)
            if op == "call_function":
                example_value = proxy.node.target(*args, **kwargs)
            elif op == "call_method":
                example_value = getattr(args[0], proxy.node.target)(*args[1:], **kwargs)
            elif op == "call_module":
                assert nnmodule is not None
                example_value = copy.deepcopy(nnmodule)(*args, **kwargs)
            else:
                assert False, op
            torch.random.set_rng_state(rng)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng)

        if isinstance(example_value, torch.Tensor):
            proxy.node.meta["example_value"] = clone_tensor(example_value)
            options.update(TensorVariable.specialize(example_value))
            return TensorVariable(proxy, **options)
        elif (
            istype(example_value, (torch.Size, int, bool, float))
            and config.dynamic_shapes
        ):
            proxy.node.meta["example_value"] = example_value
            if isinstance(example_value, torch.Size):
                options["dyn_shape_len"] = len(example_value)
            return DynamicShapeVariable(proxy, type(example_value), **options)
        elif istype(example_value, int) and proxy.node.target in (
            torch.seed,
            operator.mod,
            torch.distributed.get_rank,
            torch.distributed.get_world_size,
        ):
            proxy.node.meta["example_value"] = example_value
            return DynamicShapeVariable(proxy, type(example_value), **options)
        elif istype(example_value, torch.Size) and all(
            [isinstance(x, int) for x in example_value]
        ):
            sizes = [variables.ConstantVariable(x) for x in example_value]
            return SizeVariable(sizes, **options)
        elif isinstance(example_value, (tuple, list)):
            unpacked = []
            for i, val in enumerate(example_value):
                unpacked.append(
                    TensorVariable.create(
                        tx,
                        proxy.tracer.create_proxy(
                            "call_function", operator.getitem, (proxy, i), {}
                        ),
                        example_value=val,
                        **options,
                    )
                )
            if istype(example_value, (tuple, list)):
                return variables.TupleVariable(unpacked, **options)
            else:
                assert (
                    example_value.__class__.__module__ == "torch.return_types"
                    or hasattr(example_value, "_fields")
                ), "namedtuple?"
                return variables.NamedTupleVariable(
                    unpacked, example_value.__class__, **options
                )
        elif example_value is None or proxy.node.target is torch.manual_seed:
            return variables.ConstantVariable(None, **options)
        else:
            assert (
                False
            ), f"torch.* op returned non-Tensor {typestr(example_value)} {proxy.node.op} {proxy.node.target}"

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        dtype=None,
        device=None,
        ndim=None,
        size=None,
        stride=None,
        requires_grad=None,
        is_quantized=None,
        **kwargs,
    ):
        super(TensorVariable, self).__init__(**kwargs)
        self.proxy = proxy
        self.dtype = dtype
        self.device = device
        self.ndim = ndim
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return torch.Tensor

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {
            "dtype": value.dtype,
            "device": value.device,
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_quantized": value.is_quantized,
        }
        if not config.dynamic_shapes:
            props["size"] = tuple(value.size())
            props["stride"] = tuple(value.stride())
        return props

    def var_getattr(self, tx, name):
        from . import ConstantVariable
        from . import TorchVariable

        result = None
        options = VariableTracker.propagate(self)
        if name == "ndim" and self.ndim is not None:
            result = ConstantVariable(self.ndim, **options)
        elif name == "dtype" and self.dtype is not None:
            result = TorchVariable(self.dtype, **options)
        elif name == "device" and self.device is not None:
            result = TorchVariable(self.device, **options)
        elif name == "is_cuda" and self.device is not None:
            result = ConstantVariable(self.device.type == "cuda", **options)
        elif name == "shape" and self.size is not None:
            result = ConstantVariable(self.size, **options)
        elif name == "requires_grad" and self.requires_grad is not None:
            result = ConstantVariable(self.requires_grad, **options)
        elif name == "is_quantized" and self.is_quantized is not None:
            result = ConstantVariable(self.is_quantized, **options)
        elif name == "shape" and self.size is None:
            result = self.call_method(tx, "size", [], {})
        elif name == "ndim" and self.ndim is None:
            result = self.call_method(tx, "dim", [], {})

        if name == "__class__":
            return TorchVariable(self.python_type(), **options)

        if result is None:
            raise NotImplementedError()

        return result

    def unpack_var_sequence(self, tx):
        options = VariableTracker.propagate(self)
        if self.size:
            return [
                variables.BuiltinVariable(operator.getitem, **options).call_function(
                    tx, [self, variables.ConstantVariable(i)], {}
                )
                for i in range(self.size[0])
            ]

        return super(TensorVariable, self).unpack_var_sequence(tx)

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

        if name == "stride" and self.stride is not None:
            constant_result = ConstantVariable(self.stride, **options)
        elif name == "size" and self.size is not None:
            constant_result = ConstantVariable(self.size, **options)
        elif name == "numel" and self.size is not None:
            constant_result = ConstantVariable(product(self.size), **options)
        elif name in ("ndimension", "dim") and self.ndim is not None:
            constant_result = ConstantVariable(self.ndim, **options)
        elif name == "is_floating_point" and self.dtype is not None:
            constant_result = ConstantVariable(self.dtype.is_floating_point, **options)
        else:
            constant_result = None

        if constant_result:
            assert not kwargs
            if len(args) == 1:
                return constant_result.getitem_const(args[0])
            elif args:
                return TupleVariable(
                    [constant_result.getitem_const(a) for a in args], **options
                )
            return constant_result
        elif (
            name == "repeat"
            and not all(
                x.is_python_constant() for x in itertools.chain(args, kwargs.values())
            )
            and not config.dynamic_shapes
        ):
            unimplemented("dynamic Tensor.repeat")
        elif name in ("item", "tolist", "numpy", "backward"):
            unimplemented(f"Tensor.{name}")
        elif name == "nonzero" and not config.dynamic_shapes:
            unimplemented(f"Tensor.{name}")
        elif name == "__len__":
            if self.size:
                assert not config.dynamic_shapes
                return ConstantVariable(self.size[0], **options)
            else:
                return TensorVariable.create(
                    tx,
                    tx.output.create_proxy(
                        "call_function", len, (self.as_proxy(),), {}
                    ),
                    **options,
                )
        elif name == "__setitem__":
            tx.output.guards.update(options["guards"])
            tx.output.create_proxy(
                "call_function",
                operator.setitem,
                *proxy_args_kwargs([self] + args, kwargs),
            )
            return ConstantVariable(None, **options)
        else:
            return TensorVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
                **options,
            )


class DynamicShapeVariable(TensorVariable):
    def __init__(self, proxy, dyn_shape_cls, dyn_shape_len=None, **kwargs):
        super(DynamicShapeVariable, self).__init__(proxy, **kwargs)
        self.dyn_shape_cls = dyn_shape_cls
        self.dyn_shape_len = dyn_shape_len

    def python_type(self):
        return self.dyn_shape_cls

    def unpack_var_sequence(self, tx):
        if self.dyn_shape_len is not None:
            return [
                variables.BuiltinVariable(
                    operator.getitem, **VariableTracker.propagate(self)
                ).call_function(tx, [self, variables.ConstantVariable(i)], {})
                for i in range(self.dyn_shape_len)
            ]
        super(DynamicShapeVariable, self).unpack_var_sequence(tx)


class TensorWithTFOverrideVariable(VariableTracker):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    def __init__(
        self,
        tensor_variable,
        subclass_torch_function__func,
        subclass_type,
        **kwargs,
    ):
        super(TensorWithTFOverrideVariable, self).__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.subclass_torch_function__func = subclass_torch_function__func
        self.subclass_type = subclass_type

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        from . import GetAttrVariable

        options = VariableTracker.propagate(self, args, kwargs.values())
        # insert unwrapped version of self as the first argument
        args = list(args)
        args.insert(0, self.tensor_variable)
        func_var = GetAttrVariable(self.tensor_variable, name)

        unwrapped = TensorWithTFOverrideVariable.inline_torch_function_unwrapped(
            tx,
            func_var,
            self.subclass_torch_function__func,
            self.subclass_type,
            options,
            args,
            kwargs,
        )

        # TODO(future PR): implement rewrapping conditional on method presence
        # in `torch.overrides.get_default_nowrap_function()`. It's unclear how
        # to do this easily in the current codebase since the resolution of
        # `GetAttrVariable` depends on the type of the underlying object.

        return TensorWithTFOverrideVariable(
            unwrapped,
            self.subclass_torch_function__func,
            self.subclass_type,
        )

    @staticmethod
    def inline_torch_function_unwrapped(
        tx,
        original_func_var,
        tf_func,
        subclass_type,
        options,
        args,
        kwargs,
    ):
        """
        This function inlines the `__torch_function__` override for `original_func_var`.
        For example, if the user code is

           x1 = torch.sigmoid(x0)

        And `x0` has an override, then:
        * `original_func_var` will be a `VariableTracker` object wrapping `torch.sigmoid`
        * `tf_func` will be the custom `__torch_function__` function
        * `subclass_type` will be `type(x0)`

        The caller is expected to properly massage args and kwargs before
        passing them into this function.

        The caller is responsible for wrapping the return value, if needed.
        """
        from torchdynamo.variables import UserDefinedClassVariable
        from torchdynamo.variables import UserFunctionVariable
        from torchdynamo.variables.builder import TupleVariable

        tf_func_var = UserFunctionVariable(tf_func, **options)
        type_var = UserDefinedClassVariable(subclass_type, **options)

        # signature:
        # def __torch_function__(cls, func, types, args=(), kwargs=None):
        tf_args = (
            type_var,  # cls
            original_func_var,  # func
            (type_var,),  # types
            TupleVariable(args),  # args
            kwargs,  # kwargs
        )

        # Disable __torch_function__ here to prevent the clone of the
        # example tensor from going into the override.
        with torch._C.DisableTorchFunction():
            return tx.inline_user_function_return(tf_func_var, tf_args, {})

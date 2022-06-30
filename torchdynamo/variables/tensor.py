import contextlib
import copy
import functools
import itertools
import numbers
import operator
from contextlib import contextmanager
from typing import Dict
from typing import List

import torch.fx
import torch.random

from ..utils import fake_tensors_available

if fake_tensors_available:
    from torch._subclasses import FakeTensor
    from ..utils import wrap_to_fake_tensor
    from ..utils import deepcopy_to_fake_tensor

from torch.fx.immutable_collections import immutable_list
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch.utils._pytree import tree_map

from torchdynamo.guards import GuardBuilder

from .. import config
from .. import variables
from ..exc import TorchRuntimeError
from ..exc import unimplemented
from ..source import AttrSource
from ..utils import clone_tensor
from ..utils import is_lazy_module
from ..utils import istype
from ..utils import product
from ..utils import proxy_args_kwargs
from .base import MutableLocal
from .base import VariableTracker
from .base import typestr
from .lists import ShapeVariable
from .lists import SizeVariable


@contextmanager
def preserve_rng_state():
    rng = torch.clone(torch.random.get_rng_state())
    if torch.cuda.is_available():
        cuda_rng = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        torch.random.set_rng_state(rng)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng)


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
        "is_contiguous",
        "is_complex",
    ]

    @staticmethod
    def propagate_args_kwargs(node):
        def visit(n: torch.fx.Node):
            return n.meta["example_value"]

        return torch.fx.node.map_arg((node.args, node.kwargs), visit)

    @staticmethod
    def run_proxy(proxy, args, kwargs, nnmodule):
        op = proxy.node.op
        if op == "call_function":
            return proxy.node.target(*args, **kwargs)
        elif op == "call_method":
            return getattr(args[0], proxy.node.target)(*args[1:], **kwargs)
        elif op == "call_module":
            assert nnmodule is not None
            return nnmodule(*args, **kwargs)
        assert False, op

    @classmethod
    def create(cls, tx, proxy, example_value=None, nnmodule=None, **options):
        if "guards" in options:
            tx.output.guards.update(options["guards"])

        assert "example_value" not in proxy.node.meta
        if not config.dynamic_propagation:
            if isinstance(example_value, torch.Tensor):
                options.update(cls.specialize(example_value))
            return cls(proxy, **options)

        use_fake_tensors = fake_tensors_available and config.fake_tensor_propagation
        if use_fake_tensors:
            fake_wrapper = functools.partial(
                wrap_to_fake_tensor, fake_mode=tx.fake_mode
            )
            # python errors if the import isnt here
            from ..utils import wrap_fake_exception
        else:

            def wrap_fake_exception(func):
                return func()

        with preserve_rng_state():
            if example_value is None:
                op = proxy.node.op
                args, kwargs = cls.propagate_args_kwargs(proxy.node)
                if use_fake_tensors:
                    args = tree_map(fake_wrapper, args)
                    kwargs = tree_map(fake_wrapper, kwargs)
                    if op == "call_module" and not is_lazy_module(nnmodule):
                        nnmodule = deepcopy_to_fake_tensor(nnmodule, tx.fake_mode)

                    def context():
                        return enable_torch_dispatch_mode(tx.fake_mode)

                else:
                    context = contextlib.nullcontext
                    if op == "call_module" and not is_lazy_module(nnmodule):
                        nnmodule = copy.deepcopy(nnmodule)

                if op == "call_module" and is_lazy_module(nnmodule):
                    assert nnmodule is not None
                    # In the case of a lazy module, we want to run
                    # the pre-hooks which initialize it
                    example_value = nnmodule(*args, **kwargs)
                try:
                    with context():
                        example_value = wrap_fake_exception(
                            lambda: cls.run_proxy(proxy, args, kwargs, nnmodule)
                        )
                except RuntimeError as e:
                    raise TorchRuntimeError() from e
            else:
                if use_fake_tensors:
                    example_value = fake_wrapper(example_value)

        if isinstance(example_value, torch.Tensor):
            # tensor subclasses will not be converted to FakeTensors and need to be cloned
            if not use_fake_tensors or not isinstance(example_value, FakeTensor):
                example_value = clone_tensor(example_value)
            proxy.node.meta["example_value"] = example_value
            specialized_props = cls.specialize(example_value)
            if use_fake_tensors and isinstance(example_value, FakeTensor):
                specialized_props["class_type"] = (
                    torch.nn.Parameter
                    if isinstance(example_value, torch.nn.Parameter)
                    else torch.Tensor
                )
            options.update(specialized_props)
            return cls(proxy, **options)
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
                if val is None:
                    # nn.MultiheadAttention() can return None, see issue #175
                    unpacked.append(
                        variables.ConstantVariable(None, **options),
                    )
                else:
                    unpacked.append(
                        cls.create(
                            tx,
                            proxy.tracer.create_proxy(
                                "call_function", operator.getitem, (proxy, i), {}
                            ),
                            example_value=val,
                            **options,
                        )
                    )
            if istype(example_value, tuple):
                return variables.TupleVariable(unpacked, **options)
            elif istype(example_value, (list, immutable_list)):
                return variables.ListVariable(
                    unpacked, mutable_local=MutableLocal(), **options
                )
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
        elif (
            isinstance(example_value, int)
            and proxy.node.target is torch._utils._element_size
        ):
            proxy.node.meta["example_value"] = example_value
            return variables.ConstantVariable(example_value, **options)
        elif (
            isinstance(example_value, numbers.Number)
            and proxy.node.target == "item"
            and config.capture_scalar_outputs
        ):
            return UnspecializedPythonVariable.create(
                tx=tx,
                proxy=proxy,
                example_value=torch.tensor(example_value),
                raw_value=example_value,
                need_unwrap=False,
                **options,
            )
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
        is_contiguous=None,
        is_complex=None,
        class_type=torch.Tensor,
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
        self.is_contiguous = is_contiguous
        self.is_complex = is_complex
        self.class_type = class_type

    def as_proxy(self):
        return self.proxy

    def python_type(self):
        return self.class_type

    def call_isinstance(self, tensor_type):
        tensortype_to_dtype = {
            torch.FloatTensor: (torch.float32, torch.float),
            torch.DoubleTensor: (torch.float64, torch.double),
            torch.HalfTensor: (torch.float16, torch.half),
            torch.BFloat16Tensor: (torch.bfloat16,),
            torch.ByteTensor: (torch.uint8,),
            torch.CharTensor: (torch.int8,),
            torch.LongTensor: (torch.int64, torch.long),
            torch.IntTensor: (torch.int32, torch.int),
            torch.ShortTensor: (torch.int16, torch.short),
            torch.BoolTensor: (torch.bool,),
        }

        def check_type(ty):
            if ty not in tensortype_to_dtype:
                return self.python_type() is ty

            dtypes = tensortype_to_dtype[ty]
            return self.dtype in dtypes

        if type(tensor_type) is tuple:
            return any([check_type(ty) for ty in tensor_type])
        else:
            return check_type(tensor_type)

    @staticmethod
    def specialize(value: torch.Tensor):
        props = {
            "dtype": value.dtype,
            "device": value.device,
            "ndim": int(value.ndim),
            "requires_grad": value.requires_grad,
            "is_quantized": value.is_quantized,
            "is_complex": value.is_complex(),
            "class_type": type(value),
        }
        if not config.dynamic_shapes:
            props["size"] = tuple(value.size())
            props["stride"] = tuple(value.stride())
            props["is_contiguous"] = value.is_contiguous()
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
            sizes = [variables.ConstantVariable(x) for x in self.size]
            result = ShapeVariable(sizes, **options)
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

        # Add a guard for type matching, these guards are checked before tensor guards
        # In some cases, a <tensor>.<attr> guard can be evaluated first, and break if
        # <tensor> is later changed to another type
        if result is not None and self.source is not None:
            result = result.add_guard(self.create_guard(GuardBuilder.TYPE_MATCH))

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
            sizes = [variables.ConstantVariable(x) for x in self.size]
            constant_result = SizeVariable(sizes, **options)
        elif name == "numel" and self.size is not None:
            constant_result = ConstantVariable(product(self.size), **options)
        elif name in ("ndimension", "dim") and self.ndim is not None:
            constant_result = ConstantVariable(self.ndim, **options)
        elif name == "is_floating_point" and self.dtype is not None:
            constant_result = ConstantVariable(self.dtype.is_floating_point, **options)
        elif name == "is_contiguous" and self.is_contiguous is not None:
            constant_result = ConstantVariable(self.is_contiguous, **options)
        elif name == "is_complex" and self.is_complex is not None:
            constant_result = ConstantVariable(self.is_complex, **options)
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
        elif name in ("tolist", "numpy", "backward"):
            unimplemented(f"Tensor.{name}")
        elif name == "nonzero" and not config.dynamic_shapes:
            unimplemented(f"Tensor.{name}")
        elif name == "item":
            if config.capture_scalar_outputs:
                return self.__class__.create(
                    tx,
                    tx.output.create_proxy(
                        "call_method", "item", (self.as_proxy(),), {}
                    ),
                    **options,
                )
            else:
                unimplemented(f"Tensor.{name}")
        elif name == "__len__":
            if self.size:
                assert not config.dynamic_shapes
                return ConstantVariable(self.size[0], **options)
            else:
                return self.__class__.create(
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
            # Convert x.new(torch.Size) into x.new_empty(torch.Size),
            # as Tensor.new acts differently with a Size input versus a tuple input.
            if (
                name == "new"
                and len(args) == 1
                and isinstance(args[0], (SizeVariable, ShapeVariable))
                and not config.dynamic_shapes
            ):
                name = "new_empty"

            return self.__class__.create(
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
        orig_tensor_variable_source,
        subclass_torch_function__func,
        subclass_type,
        **kwargs,
    ):
        super(TensorWithTFOverrideVariable, self).__init__(**kwargs)
        self.tensor_variable = tensor_variable
        self.orig_tensor_variable_source = orig_tensor_variable_source
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
            self.orig_tensor_variable_source,
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
            self.orig_tensor_variable_source,
            self.subclass_torch_function__func,
            self.subclass_type,
        )

    @staticmethod
    def inline_torch_function_unwrapped(
        tx,
        original_func_var,
        tensor_with_tf_override_source,
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
        * `tensor_with_tf_override_source` will be the `Source` object from
          the original tensor override instance in the beginning of the program
        * `tf_func` will be the custom `__torch_function__` function
        * `subclass_type` will be `type(x0)`

        The caller is expected to properly massage args and kwargs before
        passing them into this function.

        The caller is responsible for wrapping the return value, if needed.
        """
        from torchdynamo.variables import UserDefinedClassVariable
        from torchdynamo.variables.builder import TupleVariable
        from torchdynamo.variables.builder import VariableBuilder

        source = AttrSource(
            AttrSource(tensor_with_tf_override_source, "__torch_function__"),
            "__func__",
        )
        tf_func_var = VariableBuilder(tx, source)(tf_func)
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


class UnspecializedNumpyVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized numpy float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", True)
        super(UnspecializedNumpyVariable, self).__init__(proxy, **kwargs)
        self.raw_value = raw_value

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value):
        # Convert a `TensorVariable` instance into an `UnspecializedNumpyVariable` instance.
        return UnspecializedNumpyVariable(
            **dict(tensor_variable.__dict__), raw_value=raw_value
        )


class UnspecializedPythonVariable(TensorVariable):
    """
    This is a 1-element tensor represents unspecialized python float/int.
    """

    def __init__(self, proxy: torch.fx.Proxy, **kwargs):
        raw_value = kwargs.pop("raw_value", True)
        need_unwrap = kwargs.pop("need_unwrap", True)
        super(UnspecializedPythonVariable, self).__init__(proxy, **kwargs)
        self.raw_value = raw_value
        self.need_unwrap = need_unwrap

    @classmethod
    def from_tensor_variable(cls, tensor_variable, raw_value, need_unwrap=True):
        # Convert a `TensorVariable` instance into an `UnspecializedPythonVariable` instance.
        return UnspecializedPythonVariable(
            **dict(tensor_variable.__dict__),
            raw_value=raw_value,
            need_unwrap=need_unwrap,
        )

"""
This module contains tools for rewriting a dynamic PyTorch program such
that the dynamic part (e.g. control flow) can be properly captured by
tracing.
The core idea is annotating all branches in the graph with unique keys,
and using a dictionary of supplemental inputs as arguments to these
local branches so that every path gets a canonical input during tracing.

For example, consider the following usage of Python if statement:
```
if pred:
    ...
    ret = a
else:
    ...
    ret = b
```

To rewrite the code to be tracable, users may use tracing_key decorator
and cond operator:

```
@control_flow.tracing_context(inputs)
def branch_true(args):
    ...
    return a

@control_flow.tracing_context(inputs)
def branch_false(args):
    ...
    return b

ret = control_flow.cond(pred, branch_true, branch_false, args)
```

This operation is not yet supported in torchdynamo tracing, but will be. 

"""

from typing import Callable, Union

import torch
import torch.utils._pytree as pytree
import torchdynamo


@torchdynamo.eval_frame._logical_handling
def _cond_live(pred, t, f):
    if pred:
        return t
    else:
        return f

@torchdynamo.eval_frame._logical_handling
def cond(pred, true_fn, false_fn, inputs):
    """
    A higher order function returning result based on passed predicate
    value and conditionally execute one of true_fn and false_fn.

    Detects whether a tracer is present in the context, and if so will
    trace_through both true_fn and false_fn with local inputs provided
    by tracing_context dictionary from the current tracer. When
    returning, wraps two traced graphs into a cond() call and construct
    a call_function node in the tracer's graph.

    Checks and enforces that the returning value(s) from both
    branches has the same Tensor type. For now enforces that both
    branches have the same number of tensor inputs.
    """
    # flattened_inputs, _ = pytree.tree_flatten(inputs)

    # if not all([isinstance(i, torch.Tensor) for i in flattened_inputs]):
    #     raise ValueError(
    #         f"control_flow.cond() expects all inputs values to be tensors, actual inputs: {inputs}"
    #     )
    # import torchdynamo
    # with torchdynamo.disable():
    if pred:
        print("What the fuck is true_fn?", true_fn)
        return true_fn(*inputs)
    return false_fn(*inputs)
    # outputs = true_fn(*inputs) if pred else false_fn(*inputs)

    # flattened_outputs, _ = pytree.tree_flatten(outputs)

    # if not all([isinstance(r, torch.Tensor) for r in flattened_outputs]):
    #     raise ValueError(
    #         f"control_flow.cond() only supports tensor as output, actual output: {outputs}"
    #     )

    return outputs

    # # Once global tracer is present, we need to assume all tensors are
    # # PythonTensor wrapped with FunctionalTensorWrapper.

    # gm_true = make_submodule(true_fn, example_returns=flattened_outputs)
    # gm_false = make_submodule(false_fn, example_returns=flattened_outputs)
    # proxies = tuple([unwrap_proxy(i) for i in flattened_inputs])

    # proxy = tracer.create_proxy(
    #     "call_function",
    #     cond,
    #     (unwrap_proxy(pred), gm_true, gm_false, proxies),
    #     {},
    # )

    # return tree_return(outputs, proxy, update_with_proxy)


def while_loop(cond_fn, body_fn, init_val):
    flattened_inputs, _ = pytree.tree_flatten(init_val)
    if not all([isinstance(i, torch.Tensor) for i in flattened_inputs]):
        raise ValueError(
            f"control_flow.while_loop() expects all inputs values to be tensors, actual inputs: {init_val}"
        )

    val = init_val
    while cond_fn(*val):
        val = body_fn(*val)

    flattened_outputs, _ = pytree.tree_flatten(val)
    if not all([isinstance(o, torch.Tensor) for o in flattened_outputs]):
        raise ValueError(
            f"control_flow.while_loop() expects all returned values to be tensors, actual outputs: {val}"
        )


    return val

    # gm_cond = make_submodule(cond_fn, single_return=True)
    # gm_body = make_submodule(body_fn)

    # proxies = tuple([unwrap_proxy(v) for v in flattened_inputs])

    # proxy = tracer.create_proxy(
    #     "call_function",
    #     while_loop,
    #     (gm_cond, gm_body, proxies),
    #     {},
    # )

    # return tree_return(val, proxy, update_with_proxy)


# def tracing_context(inputs: Union[tuple, Callable]):
#     """
#     A decorator function to annotate code path that we conditionally
#     run during tracing. We need to annotate these paths for now because
#     during exir.trace(), the tracer does not know what's the proper
#     local inputs to be passed to the untaken path.
#     """

#     def decorator(f):
#         def wrapper(*args, **kwargs):
#             if kwargs:
#                 raise ValueError(
#                     "kwargs are not supported for @tracing_context decorated functions."
#                 )

#             return f(*args)

#         tracing_inputs = inputs() if isinstance(inputs, Callable) else inputs

#         if not isinstance(tracing_inputs, tuple):
#             raise ValueError(
#                 f"tracing_context only takes tuple inputs to local branch, got type: {type(tracing_inputs)}"
#             )

#         wrapper.__tracing_inputs__ = tracing_inputs
#         return wrapper

#     return decorator

def cond(pred, true_fn, false_fn, inputs):
    """
    A conditional control flow operator, akin to if.

    Use in place of "if" to ensure both the true and false path are traced.

    pred - a condition evaluted for truthiness
    """
    if pred:
        return true_fn(*inputs)
    return false_fn(*inputs)

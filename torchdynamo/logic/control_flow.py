def cond(pred, true_fn, false_fn, true_inputs, false_inputs):
    """
    A conditional control flow operator, akin to if.

    Use in place of "if" to ensure both the true and false path are traced.

    pred - a condition evaluted for truthiness
    """
    if pred:
        return true_fn(*true_inputs)
    return false_fn(*false_inputs)

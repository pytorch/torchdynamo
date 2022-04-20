import textwrap


class MissingOperator(RuntimeError):
    def __init__(self, target, args, kwargs):
        lines = ["missing lowering/decomposition", f"target: {target}"] + [
            f"args[{i}]: {arg}" for i, arg in enumerate(args)
        ]
        if kwargs:
            lines.append(f"kwargs: {kwargs}")
        super().__init__(textwrap.indent("\n".join(lines), "  ").lstrip(" -"))

from contextlib import contextmanager
from itertools import chain
from threading import local

from torch.fx.graph import inplace_methods
from torch.fx.graph import magic_methods

threadlocal = local()


class Virtualized:
    """
    A global variable that redirects via thread local variable

    This allows us to swap in different op implementations in codegen.
    """

    def __init__(self, vname, default):
        self._key = f"__torchinductor_{vname}"
        self._default = default

    def set_handler(self, value):
        prior = self.get_handler()
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            yield
            self.set_handler(prior)

        return ctx()

    def get_handler(self):
        try:
            return getattr(threadlocal, self._key)
        except AttributeError:
            return self._default()

    def __getattr__(self, name):
        return getattr(self.get_handler(), name)


class NullHandler:
    pass


class MockHandler:
    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = list(map(str, args))
            fargs.extend(f"{k}={v}" for k, v in kwargs.items())
            return f"{name}({', '.join(fargs)})"

        return inner

    def masked(self, mask, body, other):
        return f"masked({mask}, {body()}, {other})"

    @classmethod
    def _init_cls(cls):
        def make_handler(format_string):
            @staticmethod
            def inner(*args):
                return format_string.format(*args)

            return inner

        for name, format_string in chain(
            magic_methods.items(), inplace_methods.items()
        ):
            setattr(cls, name, make_handler(format_string))


class WrapperHandler:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, item):
        return getattr(self._inner, item)


MockHandler._init_cls()

ops = Virtualized("ops", MockHandler)
graph = Virtualized("graph", NullHandler)
kernel = Virtualized("kernel", NullHandler)

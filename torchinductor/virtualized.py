from contextlib import contextmanager
from itertools import chain
from threading import local

from torch.fx.graph import inplace_methods
from torch.fx.graph import magic_methods

threadlocal = local()


class Virtualized:
    """
    A global variable that redirects via  thread local variable
    """

    def __init__(self, vname):
        self._key = f"__torchinductor_{vname}"

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
            return MockHandler()

    def __getattr__(self, name):
        return getattr(self.get_handler(), name)


class MockHandler:
    def __getattr__(self, name):
        def inner(*args, **kwargs):
            fargs = list(map(str, args))
            fargs.extend(f"{k}={v}" for k, v in kwargs.items())
            return f"{name}({', '.join(fargs)})"

        return inner

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

        cls.load = make_handler("{}[{}]")
        cls.store = make_handler("store({}[{}], {})")


MockHandler._init_cls()
prim = Virtualized("prim")
ops = Virtualized("ops")

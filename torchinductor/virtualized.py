from contextlib import contextmanager
from threading import local

threadlocal = local()


class Virtualized:
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
            return None

    def __getattr__(self, name):
        return getattr(self.get_handler(), name)


prim = Virtualized("prim")
ops = Virtualized("ops")
io = Virtualized("ops")


from . import cpp_codegen

prim.set_handler(cpp_codegen)
io.set_handler(cpp_codegen)
ops.set_handler(cpp_codegen)

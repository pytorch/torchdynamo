specializing = False


class Specializer:
    # For eager mode, blank by design
    def __enter__(self):
        pass

    def __exit__(self, _0, _1, _2):
        pass

    @staticmethod
    def enter():
        global specializing
        assert specializing is False, "Nested .specialize is not supported."
        specializing = True

    @staticmethod
    def exit():
        global specializing
        specializing = False


def specialize():
    return Specializer()

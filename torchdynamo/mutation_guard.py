import weakref


class MutationTracker:
    def __init__(self):
        self.mutation_count = 0
        self.watchers = []

    def on_mutation(self, name):
        self.mutation_count += 1
        tmp = self.watchers
        self.watchers = []
        for ref in tmp:
            guarded = ref()
            if guarded is not None:
                guarded.invalidate(ref)

    def track(self, guarded_code):
        self.watchers.append(weakref.ref(guarded_code))


def watch(obj, guarded_code):
    """invalidate guarded_code when obj is mutated"""
    original_cls = type(obj)
    if hasattr(original_cls, "___mutation_tracker"):
        tracker = original_cls.___mutation_tracker
    else:
        tracker = MutationTracker()

        class Shim(original_cls):
            def __setattr__(self, key, value):
                tracker.on_mutation(key)
                original_cls.__setattr__(self, key, value)

        Shim.__name__ = original_cls.__name__
        Shim.___mutation_tracker = tracker
        Shim.___real_type = original_cls
        object.__setattr__(obj, "__class__", Shim)

    tracker.track(guarded_code)


def real_type(obj):
    t = type(obj)
    return getattr(t, "__real_type", t)

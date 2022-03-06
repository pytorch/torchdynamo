import functools
import weakref

import torch.nn
from torch.nn import Module

from .utils import ExactWeakKeyDictionary


class MutationTracker:
    db = ExactWeakKeyDictionary()

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
    ensure_patched(type(obj))

    if obj not in MutationTracker.db:
        MutationTracker.db[obj] = MutationTracker()
    tracker = MutationTracker.db[obj]
    tracker.track(guarded_code)


def ensure_patched(cls):
    if getattr(cls, "___needs_mutation_patch", True):
        cls.___needs_mutation_patch = False
        original_setattr = cls.__setattr__

        @functools.wraps(original_setattr)
        def custom_setattr(self, key, value):
            try:
                MutationTracker.db[self].on_mutation(key)
            except KeyError:
                pass
            return original_setattr(self, key, value)

        cls.__setattr__ = custom_setattr


class GenerationTracker:
    generation = 0
    db = ExactWeakKeyDictionary()
    dynamic_classes = ExactWeakKeyDictionary()

    @classmethod
    def tag(cls, obj):
        cls.db[obj] = cls.generation

    @staticmethod
    def mark_class_dynamic(cls):
        assert issubclass(cls, torch.nn.Module)
        GenerationTracker.dynamic_classes[cls] = True

    @classmethod
    def check(cls, obj):
        return cls.db.get(obj, -1) == cls.generation


def generation_tagging_new(cls, *args, **kwargs):
    obj = object.__new__(cls)
    GenerationTracker.tag(obj)
    return obj


def is_dynamic_nn_module(obj):
    """Check for nn.Modules() created dynamically or mutated"""
    return GenerationTracker.dynamic_classes.get(type(obj)) or GenerationTracker.check(
        obj
    )


def install_generation_tagging_new():
    """
    Monkey patch torch.nn.Module.__new__ so we can detect nn.Module
    instances created dynamically inside forward methods.
    """
    assert (
        Module.__dict__.get("__new__") is generation_tagging_new
        or Module.__new__ is object.__new__
    )
    Module.__new__ = generation_tagging_new
    GenerationTracker.db.clear()
    GenerationTracker.generation += 1

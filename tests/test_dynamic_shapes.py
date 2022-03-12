#!/usr/bin/env pytest
import functools
from unittest.mock import patch

import torchdynamo

from . import test_functions
from . import test_misc
from . import test_modules
from . import test_repros


def make_dynamic_fn(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with patch.object(torchdynamo.config, "dynamic_shapes", True):
            return fn(*args, **kwargs)

    return _fn


def make_dynamic_cls(cls):
    class DynamicShapeTest(cls):
        pass

    DynamicShapeTest.__name__ = f"DynamicShapes{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue
            new_name = f"{name}_dynamic_shapes"
            fn = make_dynamic_fn(fn)
            fn.__name__ = new_name
            setattr(DynamicShapeTest, name, None)
            setattr(DynamicShapeTest, new_name, fn)

    return DynamicShapeTest


DynamicShapesFunctionTests = make_dynamic_cls(test_functions.FunctionTests)
DynamicShapesMiscTests = make_dynamic_cls(test_misc.MiscTests)
DynamicShapesReproTests = make_dynamic_cls(test_repros.ReproTests)
DynamicShapesNNModuleTests = make_dynamic_cls(test_modules.NNModuleTests)

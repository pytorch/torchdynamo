#!/usr/bin/env pytest
import functools
from unittest.mock import patch

import torchdynamo

from . import test_functions
from . import test_misc
from . import test_modules
from . import test_repros
from . import test_unspec


def make_no_fake_fn(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        with patch.object(torchdynamo.config, "fake_tensor_propagation", False):
            return fn(*args, **kwargs)

    return _fn


def make_no_fake_cls(cls):
    class NoFakeTensorsTest(cls):
        pass

    NoFakeTensorsTest.__name__ = f"NoFakeTensors{cls.__name__}"

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                continue
            new_name = f"{name}_no_fake_tensors"
            fn = make_no_fake_fn(fn)
            fn.__name__ = new_name
            setattr(NoFakeTensorsTest, name, None)
            setattr(NoFakeTensorsTest, new_name, fn)

    return NoFakeTensorsTest


NoFakeTensorsFunctionTests = make_no_fake_cls(test_functions.FunctionTests)
NoFakeTensorsMiscTests = make_no_fake_cls(test_misc.MiscTests)
NoFakeTensorsReproTests = make_no_fake_cls(test_repros.ReproTests)
NoFakeTensorsNNModuleTests = make_no_fake_cls(test_modules.NNModuleTests)
NoFakeTensorsUnspecTests = make_no_fake_cls(test_unspec.UnspecTests)

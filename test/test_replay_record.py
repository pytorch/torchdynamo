#!/usr/bin/env pytest
import logging
import re
from test.mock_modules import mock_module2

import pytest
import torch

import torchdynamo.exc as exc
import torchdynamo.testing


class ReplayRecordTests(torchdynamo.testing.TestCase):
    def check_replay(self, fn, *args, exp_exc_name=None):
        fn_opt = torchdynamo.optimize("eager")(fn)
        with self.assertLogs(logger="torchdynamo", level=logging.ERROR) as log_orig:
            try:
                fn_opt(*args)
            except:
                pass  # we'll check the logs for the raised exception

        with self.assertLogs(logger="torchdynamo", level=logging.ERROR) as log_replayed:
            file_name_match = re.search(
                r"torchdynamo\.replay\('(.*)'\)", log_orig.output[-1]
            )
            self.assertTrue(
                file_name_match is not None,
                "No record file name found in generated logs.",
            )

            torchdynamo.replay(file_name_match.groups()[0])

        def get_error_name(log):
            error_name = re.search(r"\w+Error", log.output[-1])
            self.assertIsNotNone(error_name, "No error name found in logs.")
            return error_name[0]

        orig_error = get_error_name(log_orig)
        replayed_error = get_error_name(log_replayed)
        if exp_exc_name is not None:
            self.assertEqual(orig_error, exp_exc_name)

        self.assertEqual(
            orig_error,
            replayed_error,
            "Error logs for recorded execution and replayed execution should match.",
        )

    def test_unsuccessful_inline(self):
        def level2():
            z = torch.ones(2, 2)
            a = {z: 10}  # Error here, tensor as key to dict
            return a[z] * torch.ones(1)

        def level1():
            y = torch.ones(1, 1)
            return level2()

        def level0():
            x = torch.ones(1, 1)
            level1()

        self.check_replay(level0, exp_exc_name="AssertionError")

    def test_successful_inline(self):
        def test_fn():
            x = torch.ones(2, 2)

            def level1(a):
                return a + torch.ones(2, 2)

            y = level1(x)

            return y + torch.ones(3, 3)  # dimension mismatch

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    def test_nonlocal_fn_call(self):
        def nonlocal_fn(x):
            return x + torch.ones(2, 2)

        def test_fn():
            z = torch.ones(2, 2)
            x = nonlocal_fn(z)
            return x + torch.ones(3, 3)

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    def test_nonlocal_module_fn_call(self):
        # replay when we use a module
        # not defined in the replay env
        import test.mock_modules.mock_module2

        def test_fn():
            z = test.mock_modules.mock_module2.method1([], 2)
            z = torch.ones(2, 2) + z[0]
            return z + torch.zeros(3, 3)

        self.check_replay(test_fn, exp_exc_name="RuntimeError")

    def test_nonlocal_module_class(self):
        from . import mock_modules

        def test_fn():
            z = mock_modules.mock_module2.Class1(1, 2)
            y = z.method2(torch.ones(3, 3))
            return y + torch.zeros(3, 5)

        self.check_replay(test_fn, exp_exc_name="TypeError")

    @pytest.mark.skip(reason="Need to add handling for relative imports")
    def test_local_module(self):
        def test_fn(x):
            from . import mock_modules

            z = mock_modules.mock_module3.method1([], torch.ones(5, 1))
            return torch.ones(2, 2) + x + z[0]

        self.check_replay(test_fn, torch.ones(1, 1), exp_exc_name="RuntimeError")

    # Verfiy that we replay when we have tensor arguments to the frame being replayed
    def test_fn_call_args(self):
        def test_fn(x, y):
            return x + y + torch.zeros(2, 2)

        self.check_replay(
            test_fn, torch.ones(3, 3), torch.ones(2, 2), exp_exc_name="RuntimeError"
        )

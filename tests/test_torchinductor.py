#!/usr/bin/env pytest
import contextlib
import dataclasses
import functools
import importlib
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import torchdynamo
from torchdynamo.testing import rand_strided
from torchdynamo.testing import same

try:
    importlib.import_module("functorch")
    from torchinductor import config
    from torchinductor.compile_fx import compile_fx
except (ImportError, ModuleNotFoundError):
    raise unittest.SkipTest("requires functorch")


aten = torch.ops.aten

HAS_CUDA = False
if torch.cuda.is_available():
    try:
        importlib.import_module("triton")
        HAS_CUDA = True
    except (ImportError, ModuleNotFoundError):
        pass


class TestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "debug", True))
        cls._stack.enter_context(patch.object(config.cpp, "min_chunk_size", 1))

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()


class ToTuple(torch.nn.Module):
    def forward(self, x):
        return (x,)


@dataclasses.dataclass
class InputGen:
    n: int
    device: str

    def dense(self):
        return torch.randn((self.n, self.n), device=self.device)

    def transposed(self):
        return self.dense().transpose(0, 1)

    def strided(self):
        return torch.randn((self.n * 2, self.n * 3), device=self.device)[
            self.n :, self.n :: 2
        ]

    def broadcast1(self):
        return torch.randn((self.n,), device=self.device)

    def broadcast2(self):
        return torch.randn((1, self.n, 1), device=self.device)

    def broadcast3(self):
        return torch.randn((1,), device=self.device)

    def double(self):
        return torch.randn((self.n, self.n), device=self.device, dtype=torch.double)

    def int(self):
        return torch.arange(self.n, device=self.device, dtype=torch.int32)


def check_model(self: TestCase, model, example_inputs):
    correct = model(*example_inputs)

    @torchdynamo.optimize_assert(functools.partial(compile_fx, cudagraphs=False))
    def run(*ex):
        return model(*ex)

    torchdynamo.reset()
    with unittest.mock.patch("torchdynamo.config.raise_on_backend_error", True):
        actual = run(*example_inputs)
    self.assertTrue(same(actual, correct))


def check_model_cuda(self: TestCase, model, example_inputs):
    if hasattr(model, "to"):
        model = model.to("cuda")

    def copy_fn(x):
        # preserve strides of the input on the device
        return torch.empty_strided(
            x.size(), x.stride(), device="cuda", dtype=x.dtype
        ).copy_(x)

    example_inputs = tuple(copy_fn(x) for x in example_inputs)
    check_model(self, model, example_inputs)


class SweepInputs2:
    input_gen_types1 = [
        "dense",
        "transposed",
        "strided",
        "broadcast1",
        "broadcast2",
        "broadcast3",
        "double",
        "int",
    ]
    input_gen_types2 = input_gen_types1
    gen = None

    @staticmethod
    def kernel(a, b):
        return (a + b,)

    @classmethod
    def gen_template(cls, name1, name2):
        def test(self):
            check_model(
                self,
                cls.kernel,
                (
                    getattr(cls.gen, name1)(),
                    getattr(cls.gen, name2)(),
                ),
            )

        test.__name__ = f"test_{cls.gen.device}_{name1}_{name2}"
        setattr(cls, test.__name__, test)

    @classmethod
    def populate(cls):
        for name1 in cls.input_gen_types1:
            for name2 in cls.input_gen_types2:
                cls.gen_template(name1, name2)


class SweepInputsCpuTest(SweepInputs2, TestCase):
    gen = InputGen(10, "cpu")


SweepInputsCpuTest.populate()


class CommonTemplate:
    @classmethod
    def install(my_cls, other_cls, suffix):
        for name, value in my_cls.__dict__.items():
            if name.startswith("test_"):
                setattr(other_cls, f"{name}_{suffix}", value)

    def test_add_const_int(self):
        def fn(a):
            return (a + 1,)

        self.common(fn, (torch.randn(32),))

    def test_add_const_float(self):
        def fn(a):
            return (a + 1.5,)

        self.common(fn, (torch.randn(32),))

    def test_abs(self):
        def fn(a):
            return (a / (torch.abs(a) + 1),)

        self.common(fn, (torch.randn(17),))

    def test_max_min(self):
        def fn(a, b):
            return (torch.maximum(a, b), torch.minimum(a, b))

        self.common(fn, (torch.randn(8), torch.randn(8)))

    def test_horizonal_fusion1(self):
        def fn(a, b, c):
            return (a + b, a - c, b * c)

        self.common(
            fn, (torch.randn(8, 16, 16), torch.randn(8, 16, 16), torch.randn(1, 16, 1))
        )

    def test_horizonal_fusion2(self):
        def fn(a, b, c):
            return a + 1, b + 2, c + 3

        self.common(fn, (torch.randn(8, 16, 8), torch.randn(8, 16), torch.randn(16, 8)))

    def test_sum1(self):
        def fn(a, b):
            return ((a + b).sum(-1),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sum2(self):
        def fn(a, b):
            return ((a + b).sum([1, 2]), (a + b).sum(-1))

        self.common(fn, (torch.randn(8, 9, 3, 21), torch.randn(8, 9, 3, 21)))

    def test_sum3(self):
        def fn(a, b):
            r1 = a + b
            r2 = r1.sum(-1)
            r3 = torch.squeeze(b) + 10
            return (r1, r2, r3)

        self.common(fn, (torch.randn(10, 10), torch.randn(1, 10)))

    def test_sum4(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f, e, d, c, b)

        self.common(fn, (torch.randn(1, 16, 8, 8),))

    def test_sum5(self):
        def fn(a):
            b = a + 1
            c = b.sum(-1)
            d = c + 3
            e = d.sum(-1)
            f = e + 5
            return (f,)

        self.common(fn, (torch.randn(1, 17, 8, 9),))

    def test_min_max_reduction(self):
        def fn(a, b):
            return ((a + b).max(), (a + b).min())

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_clamp(self):
        def fn(a, b):
            return (a.clamp(-0.1, 0.1), b.clamp(0), torch.clamp(a + b, max=0))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_arange(self):
        def fn(x):
            rng1 = torch.arange(8 * 8, dtype=torch.float32, device=x.device).view(8, 8)
            rng2 = torch.arange(10, 18, device=x.device)
            tmp = x * rng1
            return tmp, tmp + rng2

        self.common(fn, (torch.randn(8, 8),))

    def test_views1(self):
        def fn1(x, y):
            return (x.view(size2) + y,)

        def fn2(x, y):
            return ((x + 1).view(size2) + y,)

        views = [
            ([5 * 7], [5, 7]),
            ([2 * 3 * 4 * 5 * 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([2 * 3, 4, 5, 6 * 7], [2, 3, 4, 5, 6, 7]),
            ([10 * 5, 20], [10, 5, 20]),
            ([1, 10, 1], [10]),
            ([10, 1, 10, 1, 10], [10, 100]),
            ([2, 2, 2, 2], [4, 4]),
        ]
        for size1, size2 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

        for size2, size1 in views:
            self.common(fn1, (torch.randn(size1), torch.randn(size2)))
            self.common(fn2, (torch.randn(size1), torch.randn(size2)))

    def test_views2(self):
        def fn1(x):
            return (x.view(size2) + 1,)

        def fn2(x):
            return ((x * 2).view(size2) + 1,)

        for size1, size2 in [
            ([2, 2, 2, 2], [4, -1]),
            ([10, 1, 10, 1, 10], [-1, 100]),
            ([10 * 5, 20], [10, -1, 20]),
        ]:
            self.common(fn1, (torch.randn(size1),))
            self.common(fn2, (torch.randn(size1),))

    def test_relu(self):
        def fn(a, b):
            return (torch.relu(a), torch.relu(a + b) / 10)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_exp(self):
        def fn(a, b):
            return (torch.exp(a), torch.exp(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_sigmoid(self):
        def fn(a, b):
            return (torch.sigmoid(a), torch.sigmoid(a + b))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_silu(self):
        def fn(a):
            return (torch.nn.functional.silu(a),)

        self.common(fn, (torch.randn(8, 8),))

    def test_sum_keepdims(self):
        def fn(a, b):
            return (torch.sum(a + b, -1, keepdim=True),)

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_softmax(self):
        def fn(a, b):
            return (torch.softmax(a + b, -1), torch.softmax(a, 0), torch.softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_log_softmax(self):
        def fn(a, b):
            return (F.log_softmax(a + b, -1), F.log_softmax(a, 0), F.log_softmax(b, 1))

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_transpose(self):
        def fn(a, b):
            return (
                torch.t(a) + b,
                torch.transpose(b * 2, 0, 1) + 10,
            )

        self.common(fn, (torch.randn(8, 8), torch.randn(8, 8)))

    def test_permute(self):
        def fn(a):
            return (
                torch.permute(a + 1, [2, 1, 4, 0, 3]) + 2,
                torch.permute(a, [2, 1, 4, 0, 3]) + 2,
            )

        self.common(fn, (torch.randn(2, 2, 2, 2, 2),))

    def test_expand(self):
        def fn(a):
            return ((a + 1).expand(3, 4, 2, 3, 2) + 2, a.expand(2, 1, 2, 3, 2) + 2)

        self.common(fn, (torch.randn(2, 1, 2),))

    def test_squeeze1(self):
        def fn(a):
            return ((a + 1).squeeze() + 2, a.squeeze() + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 1, 1),))

    def test_squeeze2(self):
        def fn(a):
            return ((a + 1).squeeze(-1).squeeze(2) + 2, a.squeeze(0) + 2)

        self.common(fn, (torch.randn(1, 2, 1, 2, 2, 2, 1),))

    def test_unsqueeze(self):
        def fn(a):
            return (
                torch.unsqueeze(a + 1, -1) + 2,
                torch.unsqueeze(a, 2) + 2,
                torch.unsqueeze(a + 1, 0) + 2,
                torch.unsqueeze(a, -2) + 2,
            )

        self.common(
            fn,
            (
                torch.randn(
                    2,
                    2,
                    2,
                    2,
                ),
            ),
        )

    def test_addmm(self):
        def fn(a, b, c):
            return (torch.addmm(a + 1, b + 2, c + 3) + 4,)

        self.common(
            fn,
            (
                torch.randn(8, 8),
                torch.randn(8, 8),
                torch.randn(8, 8),
            ),
        )

    def test_linear(self):
        mod = torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Sigmoid(),
            ToTuple(),
        )
        self.common(mod, (torch.randn(2, 8),))

    def test_bmm(self):
        def fn(a, b):
            return (
                torch.bmm(a, b),
                torch.bmm(a + 1, b + 2) + 3,
            )

        self.common(
            fn,
            (
                torch.randn(2, 8, 8),
                torch.randn(2, 8, 8),
            ),
        )
        self.common(
            fn,
            (
                torch.randn(1, 16, 8),
                torch.randn(1, 8, 10),
            ),
        )

    def test_gather(self):
        def fn(a, b):
            return (torch.gather(a.expand([4, 5, 10, 6]), 3, b + 1),)

        self.common(
            fn,
            (
                torch.randn([1, 1, 10, 6]),
                torch.randint(5, [4, 5, 10, 1], dtype=torch.int64),
            ),
        )

    def test_slice1(self):
        def fn(a):
            return (
                a[:, :10, 0] + a[:, 10:, 0],
                (a + 1)[:, :10, 0] + (a + 1)[:, 10:, 0],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_slice2(self):
        def fn(a):
            return (
                a[:-1, ::2, -1] + a[-1:, 1::2, -2],
                (a + 1)[:-1, ::2, -1] + (a + 2)[-1:, 1::2, -2],
            )

        self.common(
            fn,
            (torch.randn([2, 20, 2]),),
        )

    def test_split(self):
        def fn(a):
            t = torch.split(a, 3, -1)
            return (t[0], t[1], t[2], t[3])

        def fn2(a):
            return fn(a + 1)

        self.common(
            fn,
            (torch.randn([2, 2, 10]),),
        )

        self.common(
            fn2,
            (torch.randn([2, 2, 10]),),
        )

    def test_convolution(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, [3, 3]),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randn([2, 4, 16, 16]),),
        )

    def test_adaptive_avg_pool2d1(self):
        def fn(x):
            return aten._adaptive_avg_pool2d(x, (6, 6)), aten._adaptive_avg_pool2d(
                x + 1, (4, 5)
            )

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_max_pool2d1(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn(2, 4, 16, 16),),
        )

    def test_max_pool2d2(self):
        def fn(x):
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2])

        self.common(
            fn,
            (torch.randn([16, 64, 55, 55]),),
        )

    def test_max_pool2d3(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [1, 1])

        self.common(
            fn,
            (-torch.arange(1 * 8 * 8, dtype=torch.float32).view(1, 1, 8, 8),),
        )

    def test_max_pool2d4(self):
        def fn(x):
            # with padding
            return aten.max_pool2d_with_indices(x, [3, 3], [2, 2], [0, 0], [1, 1], True)

        self.common(
            fn,
            (torch.randn([2, 8, 111, 111]),),
        )

    def test_alexnet_prefix(self):
        def forward(arg6, arg7, arg16):
            convolution = torch.ops.aten.convolution(
                arg16, arg7, arg6, [4, 4], [2, 2], [1, 1], False, [0, 0], 1
            )
            relu = torch.ops.aten.relu(convolution)
            max_pool2d_with_indices = torch.ops.aten.max_pool2d_with_indices(
                relu, [3, 3], [2, 2]
            )
            getitem = max_pool2d_with_indices[0]
            return (getitem,)

        self.common(
            forward,
            (
                rand_strided((64,), (1,), torch.float32, "cpu"),
                rand_strided((64, 3, 11, 11), (363, 121, 11, 1), torch.float32, "cpu"),
                rand_strided(
                    (16, 3, 224, 224), (150528, 50176, 224, 1), torch.float32, "cpu"
                ),
            ),
        )

    def test_elu(self):
        def fn(x):
            return aten.elu(x, 1.6732632423543772, 1.0507009873554805) + 2, aten.elu(
                x + 1, 2, 3, 4
            )

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_tanh(self):
        def fn(x):
            return aten.tanh(x) + 2, aten.tanh(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_repeat(self):
        def fn(x):
            return (
                x.repeat(2, 2, 3, 1),
                x.repeat(8, 1, 1, 1),
                x.repeat(2, 1, 1, 1, 1, 1),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_embedding(self):
        m = torch.nn.Sequential(
            torch.nn.Embedding(10, 4, padding_idx=0),
            torch.nn.ReLU(),
            ToTuple(),
        )

        self.common(
            m,
            (torch.randint(10, [2, 8]),),
        )

    def test_mean(self):
        def fn(x):
            return (
                x.mean(),
                x.mean(-1),
                torch.mean(x, -2, keepdim=True),
                x.mean([0, 1]),
            )

        self.common(
            fn,
            (torch.randn([1, 2, 4, 8]),),
        )

    def test_std(self):
        def fn(x):
            return (
                torch.var(x, True),
                torch.var(x, False),
                torch.var(x, -1, True),
                torch.var(x, -1, False),
                torch.std(x, False),
                torch.std(x, [0, 1], True),
                torch.std(x, [0, 1], False),
                torch.std(x, -2, True, keepdim=True),
            )

        self.common(
            fn,
            (torch.randn([2, 4, 4, 8]),),
        )

    def test_embedding_bag(self):
        def fn(w, i, o):
            return aten._embedding_bag(w, i, o, False, 0, False, None)

        self.common(
            fn,
            (torch.randn([10, 4]), torch.randint(10, [8]), torch.tensor([0, 2, 6])),
        )

    def test_batch_norm_2d(self):
        m = torch.nn.Sequential(
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU(),
        )
        m.eval()
        self.common(
            m,
            (torch.randn([2, 10, 8, 8]),),
        )
        self.common(
            m,
            (torch.randn([3, 10, 16, 16]),),
        )

    def test_leaky_relu(self):
        def fn(x):
            return aten.leaky_relu(x, 0.2) + 2, aten.leaky_relu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_gelu(self):
        def fn(x):
            return aten.gelu(x) + 2, aten.gelu(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_clone(self):
        def fn(x):
            return aten.clone(x) + 2, aten.clone(x + 1)

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_masked_fill(self):
        def fn(mask, value):
            return aten.masked_fill(value, mask, -1000000000.0) + 2, aten.masked_fill(
                value / 2.0, torch.logical_not(mask), 667
            )

        self.common(
            fn,
            (
                torch.randint(0, 1, [1, 16], dtype=torch.bool),
                torch.randn([16, 16]),
            ),
        )

    def test_pow(self):
        def fn(x):
            return [aten.pow(x, e) for e in range(-8, 9)]

        self.common(
            fn,
            (torch.randn([16, 16]),),
        )

    def test_glu(self):
        def fn(x):
            return aten.glu(x, -1), aten.glu(x, 1), aten.glu(x, 2)

        self.common(
            fn,
            (torch.randn([8, 16, 8, 8]),),
        )

    def test_cat(self):
        def fn(a):
            tmp = a * 2
            return torch.cat((a, a[:, :4] + 1, a + 2), -1), torch.cat((tmp, tmp), 0)

        self.common(
            fn,
            (torch.randn([8, 16]),),
        )


class CpuTests(TestCase):
    common = check_model


CommonTemplate.install(CpuTests, "cpu")

if HAS_CUDA:

    class SweepInputsCudaTest(SweepInputs2, TestCase):
        gen = InputGen(10, "cuda")

    SweepInputsCudaTest.populate()

    class GpuTests(TestCase):
        common = check_model_cuda

        def test_simplify_dims(self):
            def fn(a):
                return (a + 1,)

            self.common(
                fn, (torch.randn(2, 3, 10, 5, 6, device="cuda")[:, :, 2::2, :, :],)
            )

    CommonTemplate.install(GpuTests, "cuda")

#!/usr/bin/env pytest
from ptdynamo import optimizer
import unittest

c = 10


def fn1(a, b):
    return a + b - c


def fn2(a, b):
    x = 0
    y = 1

    def modify():
        nonlocal x
        x += a + b + c

    for _ in range(2):
        modify()

    return x + y


def fn3():
    yield 1
    yield 2


with_debug_nops = optimizer.context(optimizer.debug_insert_nops)


class NopTests(unittest.TestCase):
    @with_debug_nops
    def test1(self):
        self.assertEqual(fn1(1, 2), -7)
        self.assertEqual(fn1(1, 2), -7)

    @with_debug_nops
    def test2(self):
        self.assertEqual(fn2(1, 2), 27)
        self.assertEqual(fn2(1, 2), 27)

    @with_debug_nops
    def test3(self):
        t = fn3()
        self.assertEqual(next(t), 1)
        self.assertEqual(next(t), 2)
        self.assertRaises(StopIteration, lambda: next(t))


class AssemblerTests(unittest.TestCase):


strip_extended_args(instructions)

from ptdynamo import optimizer

import logging

logging.basicConfig(level=logging.INFO)

c = 10


def test2(a, b):
    x = 0
    y = 1

    def modify():
        nonlocal x
        x += a + b + c

    for _ in range(2):
        modify()

    return x + y


def test1(a, b):
    return a + b - c


def foo():
    yield 1
    yield 2


with optimizer.context():
    print(-7, "==", test1(1, 2))
    print(27, "==", test2(1, 2))
    print(-7, "==", test1(1, 2))
    print(27, "==", test2(1, 2))
    t = foo()
    print(next(t))
    print(next(t))

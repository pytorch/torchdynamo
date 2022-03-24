import collections
from typing import Dict

from sympy import Expr
from sympy import Integer
from sympy import Symbol


class SizeVarAllocator(object):
    def __init__(self, prefix="s", zero_one_const=True):
        super().__init__()
        self.prefix = prefix
        self.val_to_var: Dict[int, Expr] = {0: Integer(0), 1: Integer(1)}
        self.var_to_val: Dict[Expr, int] = collections.OrderedDict()
        if not zero_one_const:
            self.val_to_var.clear()

    def __getitem__(self, val):
        if val in self.val_to_var:
            return self.val_to_var[val]
        var = Symbol(f"{self.prefix}{len(self.var_to_val)}")
        self.val_to_var[val] = var
        self.var_to_val[var] = val
        return var

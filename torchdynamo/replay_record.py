import dataclasses
from dataclasses import field
from dis import Instruction
from types import CodeType
from typing import Any
from typing import List

import dill


@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    instrs: List[Instruction] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)

    def dump(self, f):
        dill.dump(self, f)

    @classmethod
    def load(cls, f):
        return dill.load(f)

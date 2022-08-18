import dataclasses
import pickle
from dataclasses import field
from dis import Instruction
from types import CodeType
from typing import Any
from typing import List


@dataclasses.dataclass
class ExecutionRecord:
    instrs: List[Instruction] = field(default_factory=list)
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)

    @property
    def code_options(self):
        return self._code_options

    # NB: We can't serialize the code object (we could with marshal)
    # but we don't need it since we have the instructions anyway
    @code_options.setter
    def code_options(self, opts):
        self._code_options = {
            k: v for k, v in opts.items() if not isinstance(v, CodeType)
        }

    def dump(self, f):
        assert not any(
            [isinstance(v, CodeType) for v in self.code_options.values()]
        ), "We can't serialize the code type"
        pickle.dump(self, f)

    @classmethod
    def load(cls, f):
        return pickle.load(f)

import dataclasses
import enum
import types
from typing import Optional, Set, List


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1


class GuardRequirement(enum.Enum):
    TYPE_MATCH = 0
    VALUE_MATCH = 1
    FUNCTION_MATCH = 2  # e.q. "from torch import add"


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    requirement: GuardRequirement

    def __hash__(self):
        return hash((self.name, self.source, self.requirement))


class GuardedCode:
    def __init__(self, code: types.CodeType, guards: Optional[Set[Guard]] = None):
        self.code = code
        self.value_locals: List[str] = []
        self.value_globals: List[str] = []
        self.type_locals: List[str] = []
        self.type_globals: List[str] = []
        for guard in (guards or []):
            if guard.requirement == GuardRequirement.VALUE_MATCH:
                if guard.source == GuardSource.LOCAL:
                    self.value_locals.append(guard.name)
                else:
                    assert guard.source == GuardSource.GLOBAL
                    self.value_globals.append(guard.name)
            elif guard.requirement == GuardRequirement.TYPE_MATCH:
                if guard.source == GuardSource.LOCAL:
                    self.type_locals.append(guard.name)
                else:
                    assert guard.source == GuardSource.GLOBAL
                    self.type_globals.append(guard.name)
            else:
                assert guard.requirement == GuardRequirement.FUNCTION_MATCH
                # TODO(jansel): may want to guard "from torch import add" in the future
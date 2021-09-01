import dataclasses
import enum


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

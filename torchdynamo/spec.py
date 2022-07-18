from enum import Enum
from enum import unique

import torch


# TODO(voz): Spec today offers no reconstrution methods.
# This means we can describe the spec, but not turn a flat list of Tensors into a materialized spec.
class Spec:
    @staticmethod
    def describe_spec(t):
        spec = Spec()

        # TODO(voz): Consolidate w/ the Element enum below, strings is fine for now, but we want to
        # properly assemble a Spec
        def _type_to_element_open(x):
            return {
                "list": Spec.Element.OPEN_LIST,
                "tuple": Spec.Element.OPEN_TUPLE,
                "dict": Spec.Element.OPEN_DICT,
            }[x.__class__.__name__]

        def _type_to_element_close(x):
            return {
                "list": Spec.Element.CLOSE_LIST,
                "tuple": Spec.Element.CLOSE_TUPLE,
                "dict": Spec.Element.CLOSE_DICT,
            }[x.__class__.__name__]

        def _describe_spec(t):
            nonlocal spec
            if isinstance(t, torch.Tensor):
                spec.add_element(Spec.Element.TENSOR)
            if isinstance(t, (list, set, tuple)):
                spec.add_element(_type_to_element_open(t))
                for e in t:
                    _describe_spec(e)
                spec.add_element(_type_to_element_close(t))
            if isinstance(t, dict):
                spec.add_element(_type_to_element_open(t))
                for k, v in t:
                    # TODO(voz): Handle keys properly, this has a chance of causing a false-negative on export match.
                    _describe_spec(v)
                spec.add_element(_type_to_element_close(t))

        _describe_spec(t)
        return spec

    def __init__(self):
        self.elements = []

    @unique
    class Element(str, Enum):
        TENSOR = 0
        OPEN_LIST = 1
        CLOSE_LIST = 2
        OPEN_TUPLE = 3
        CLOSE_TUPLE = 4
        OPEN_DICT = 5
        CLOSE_DICT = 6
        # TODO(voz): Exhaustively support all spec elements

        @classmethod
        def _repr_dict(self):
            return {
                Spec.Element.TENSOR: "T",
                Spec.Element.OPEN_LIST: "List[",
                Spec.Element.CLOSE_LIST: "]",
                Spec.Element.OPEN_TUPLE: "Tuple(",
                Spec.Element.CLOSE_TUPLE: ")",
                Spec.Element.OPEN_DICT: "Dict{",
                Spec.Element.CLOSE_DICT: "}",
            }

        def __repr__(self):
            return self._repr_dict()[self]

    def add_element(self, element: Element):
        # TODO(voz): Add validation logic, (ex: no list close if no list opens on the stack)
        self.elements.append(element)

    def __repr__(self):
        rep = ""
        for x in self.elements:
            rep += f"{x!r}"
        return rep


from enum import Enum, unique

class Spec():
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
        #TODO(voz): Exhaustively support all spec elements

        @classmethod
        def _repr_dict(self):
            return {
                Spec.Element.TENSOR : "T",
                Spec.Element.OPEN_LIST : "List[",
                Spec.Element.CLOSE_LIST : "]",
                Spec.Element.OPEN_TUPLE : "Tuple(",
                Spec.Element.CLOSE_TUPLE : ")",
                Spec.Element.OPEN_DICT : "Dict{",
                Spec.Element.CLOSE_DICT : "}",
            }

        def __repr__(self):
            return self._repr_dict()[self]


    def add_element(self, element: Element):
        #TODO(voz): Add validation logic, (ex: no list close if no list opens on the stack)
        self.elements.append(element)


    def __repr__(self):
        rep = ""
        for x in self.elements:
            rep += f"{x!r}"
        return rep
    
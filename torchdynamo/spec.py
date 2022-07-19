from enum import Enum
from enum import unique
from typing import List

import torch


class Spec:
    @staticmethod
    def describe_spec(t):
        spec = Spec()

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
        self.tensors = 0

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
        # TODO(voz): Add validation logic, (ex: no list close if no list opens on the stack).
        # It currently exists in apply, but not here.
        self.elements.append(element)
        if element == Spec.Element.TENSOR:
            self.tensors += 1

    def __repr__(self):
        rep = ""
        for x in self.elements:
            rep += f"{x!r}"
        return rep

    @staticmethod
    def apply_spec(spec, tensors: List[torch.Tensor]):
        r"""
        Given a spec and a list of tensors, return a fully materialized data structure containing
        all the Tensors.

        Note: If the spec starts with a collection, the result will mirror it. Specs of only tensors, or starting
        With tensors will be added to a Tuple
        """
        if len(tensors) == 0:
            return []

        assert spec.tensors == len(
            tensors
        ), f"Spec length {spec.elements} does not match tensor list length {len(tensors)}"

        reversed_elements = spec.elements[::-1]
        collection_stack = []
        current_collection = None
        idx = 0

        def _append(e, collection):
            if isinstance(collection, list):
                collection.append(e)
            elif isinstance(collection, tuple):
                # TODO(voz): Optimize with forward scanning, this isnt that cheap
                collection += (e,)
            else:
                assert False, f"Unsupported collection type {collection.__class__}"

        result = None
        current_element = reversed_elements.pop()
        while current_element is not None:
            if current_element == Spec.Element.TENSOR:
                if current_collection is None:
                    # No current collection, make one
                    current_collection = (tensors[idx],)
                else:
                    # Add tensor to current collection
                    _append(tensors[idx], current_collection)
                # advance tensor pointer
                idx += 1
                # advance instruction
            elif current_element == Spec.Element.OPEN_LIST:
                # Store current collection away onto the stack
                collection_stack.append(current_collection)
                # Start a new list
                current_collection = list()
            elif current_element == Spec.Element.OPEN_TUPLE:
                # Store current collection away onto the stack
                collection_stack.append(current_collection)
                # Start a new tuple
                current_collection = tuple()
            elif current_element in {Spec.Element.CLOSE_LIST, Spec.Element.CLOSE_TUPLE}:
                # Closing a list is predicated on there being an open list
                assert isinstance(
                    current_collection, (list, tuple)
                ), f"Illegal spec. This is a bug in torchdynamo/spec.py. {spec}"
                # Take the last collection, we will put this list on it, if its there
                prior = collection_stack.pop()
                if prior is not None:
                    # A prior collection exists
                    _append(current_collection, prior)
                    # Now that this list is closed, we can have further additions to the prior collection
                    # Which is the collection this collection was nested in
                    current_collection = prior
                else:
                    # Closing with nothing in the stack, this is the end
                    result = current_collection
                    current_collection = None
            else:
                assert False, f"{current_element} is not yet supported"

            if len(reversed_elements) > 0:
                current_element = reversed_elements.pop()
            else:
                current_element = None
        if (current_element is not None and idx == len(tensors)) or (
            current_element is None and idx < len(tensors)
        ):
            # Note: This is checked in a precondition, but it does not hurt to ensure it here too.
            assert (
                False
            ), f"Element count in spec and number of tensors must match! Current element: {current_element}, idx: {idx}"

        return result

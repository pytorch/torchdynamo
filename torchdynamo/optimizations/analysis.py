import collections
import copy
import itertools

import torch
from torch.fx.node import map_aggregate
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes.shape_prop import _extract_tensor_metadata


class ShapeAliasingAndMutationProp(ShapeProp):
    def __init__(self, *args, **kwargs):
        super(ShapeAliasingAndMutationProp, self).__init__(*args, **kwargs)
        self.input_alias_groups = set()
        self.data_ptr_to_alias_group = dict()
        self.storage_keepalive = []
        self.make_alias_group = itertools.count(1)

    def tensor_alias_group(self, value: torch.Tensor):
        """Assign a unique identifier to the storage of a given tensor"""
        storage_data_ptr = value.storage().data_ptr()
        alias_group = self.data_ptr_to_alias_group.get(storage_data_ptr)
        if alias_group is None:
            alias_group = next(self.make_alias_group)
            self.data_ptr_to_alias_group[storage_data_ptr] = alias_group
            self.storage_keepalive.append(value.storage())
        return alias_group

    def placeholder(self, target, args, kwargs):
        value = super().placeholder(target, args, kwargs)
        assert isinstance(value, torch.Tensor)
        self.input_alias_groups.add(self.tensor_alias_group(value))
        return value

    def run_node(self, n: torch.fx.Node):
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        tensor_args = self.extract_tensors((args, kwargs))

        versions1 = [obj._version for obj in tensor_args]
        result = getattr(self, n.op)(n.target, args, kwargs)
        versions2 = [obj._version for obj in tensor_args]

        input_alias_groups = set()

        def visit_arg(arg: torch.fx.Node):
            input_alias_groups.update(arg.meta["alias_groups"])

        torch.fx.map_arg((n.args, n.kwargs), visit_arg)

        n.meta["type"] = type(result)
        n.meta["alias_groups"] = {
            self.tensor_alias_group(obj) for obj in self.extract_tensors(result)
        }
        n.meta["input_alias_groups"] = input_alias_groups
        n.meta["mutates_alias_groups"] = {
            self.tensor_alias_group(tensor)
            for tensor, v1, v2 in zip(tensor_args, versions1, versions2)
            if v1 != v2
        }
        n.meta["is_input_alias"] = bool(
            self.input_alias_groups & n.meta["alias_groups"]
        )
        n.meta["is_input_mutation"] = bool(
            self.input_alias_groups & n.meta["mutates_alias_groups"]
        )
        n.meta["is_mutation"] = bool(n.meta["mutates_alias_groups"])
        n.meta["tensor_metas"] = [
            _extract_tensor_metadata(obj) for obj in self.extract_tensors(result)
        ]
        tensors = self.extract_tensors(result)
        if tensors:
            n.meta["device"] = tensors[0].device
            n.meta["dtype"] = tensors[0].dtype
        return result

    @staticmethod
    def extract_tensors(result):
        """Return a flat list of tensors found in some nested data structure"""
        seen = set()
        tensors = []

        def visit(obj):
            if isinstance(obj, torch.Tensor) and id(obj) not in seen:
                seen.add(id(obj))
                tensors.append(obj)

        map_aggregate(result, visit)
        return tensors

    def tag_indirect_mutation(self):
        checks = collections.defaultdict(set)
        for n in self.module.graph.nodes:

            def visit_arg(arg: torch.fx.Node):
                for group in arg.meta["alias_groups"]:
                    if group in checks:
                        for other_node in checks[group]:
                            if other_node is not arg:
                                other_node.meta["indirect_mutation"] = True

            torch.fx.map_arg((n.args, n.kwargs), visit_arg)
            n.meta["indirect_mutation"] = False
            for group in n.meta["mutates_alias_groups"]:
                checks[group].add(n)

    def run(self, *args):
        try:
            super().run(*args)
            self.tag_indirect_mutation()
        finally:
            # cleanup
            self.storage_keepalive.clear()
            self.env.clear()


def has_mutation(gm, example_inputs):
    """Check if the graph module has any form of mutation"""
    # TODO - moco gives bad accuracy with Aliasing. gm is getting mutated in a bad way.
    new_gm = copy.deepcopy(gm)
    ShapeAliasingAndMutationProp(new_gm).run(*example_inputs)

    for node in new_gm.graph.nodes:
        if node.meta["is_mutation"] or node.meta["is_input_mutation"]:
            return True
    return False

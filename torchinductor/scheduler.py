import collections
import contextlib
import dataclasses
from typing import Any
from typing import Dict

from torchinductor import ir
from torchinductor.virtualized import graph


class SchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group):
        assert isinstance(node, ir.ComputedBuffer)
        self.scheduler = scheduler
        self.node = node
        self.read_writes = node.get_read_writes()
        self.group = group
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def mark_fusable(self):
        self.scheduler.fusable_deps.update(self.read_writes.writes)

    def get_name(self):
        return self.node.name

    def get_ranges(self):
        node = self.node.data
        return node.get_size(), node.get_reduction_size()

    def is_reduction(self):
        return bool(self.node.data.get_reduction_type())

    def run(self, vars, reduction_vars):
        vars = tuple(vars)
        reduction_vars = tuple(reduction_vars)
        self.scheduler.run_count += 1
        name = self.get_name()
        indexer = self.node.layout.make_indexer()
        if self.is_reduction():
            self.node.data.store_reduction(name, indexer, vars, reduction_vars)
        else:
            self.node.data.store_output(name, indexer, vars + reduction_vars)
        self.scheduler.pending_buffer_names.add(self.get_name())


@dataclasses.dataclass
class SchedulerNodeBox:
    """Allow us to invalidate a blocked node"""

    value: SchedulerNode

    def __bool__(self):
        return self.value is not None

    def pop(self) -> SchedulerNode:
        assert self
        val = self.value
        self.value = None
        return val

    def peek(self) -> SchedulerNode:
        return self.value


class BlockedNodes:
    def __init__(self):
        super().__init__()
        self.name_to_nodes = collections.defaultdict(list)
        self.dep_to_nodes = collections.defaultdict(list)

    def add(self, node: SchedulerNode):
        box = SchedulerNodeBox(node)
        for dep in node.unmet_dependencies:
            self.name_to_nodes[dep.name].append(box)
            self.dep_to_nodes[dep].append(box)

    def pop_name(self, name):
        return [x.pop() for x in self.name_to_nodes.pop(name, []) if x]

    def pop_fusable(self, deps, group):
        assert isinstance(deps, set)
        result = []
        for dep in deps:
            self.dep_to_nodes[dep] = [x for x in self.dep_to_nodes[dep] if x]
            for box in self.dep_to_nodes[dep]:
                if (
                    len(box.peek().unmet_dependencies - deps) == 0
                    and box.peek().group == group
                ):
                    result.append(box.pop())
        return result


class Scheduler:
    def __init__(self, group_fn, nodes=None):
        super(Scheduler, self).__init__()
        self.group_fn = group_fn
        self.runable_reduction_groups = collections.defaultdict(int)
        self.runable_pointwise_groups = collections.defaultdict(int)
        self.runable_nodes: Dict[Any, SchedulerNode] = collections.defaultdict(list)
        self.blocked_nodes = BlockedNodes()
        self.run_count = 0
        self.nodes = []
        self.kernels = []
        self.available_buffer_names = set(graph.graph_inputs.keys())
        self.pending_buffer_names = set()
        self.fusable_deps = set()
        if nodes:
            self.add_nodes(nodes)

    def add_nodes(self, nodes):
        for node in nodes:
            group = (
                self.group_fn(node.data.get_size()),
                self.group_fn(node.data.get_reduction_size()),
            )
            self.nodes.append(SchedulerNode(self, node, group))
        self.enqueue(self.nodes)

    def enqueue(self, node):
        if isinstance(node, (tuple, list)):
            for n in node:
                self.enqueue(n)
            return

        assert isinstance(node, SchedulerNode)
        if node.unmet_dependencies:
            self.blocked_nodes.add(node)
        else:
            self.runable_nodes[node.group].append(node)
            if node.is_reduction():
                # Do reductions first as they yield more possible fusions
                self.runable_reduction_groups[node.group] += 1
            else:
                self.runable_pointwise_groups[node.group] += 1

    def barrier(self):
        """
        Mark all pending_buffer_names as available and enqueue any nodes
        that became runable.
        """
        self.available_buffer_names.update(self.pending_buffer_names)
        nodes_to_add = []
        for name in self.pending_buffer_names:
            for node in self.blocked_nodes.pop_name(name):
                node.prune_deps()
                nodes_to_add.append(node)
        self.pending_buffer_names.clear()
        self.enqueue(nodes_to_add)

    def kernel(self, kernel):
        self.fusable_deps.clear()
        self.kernels.append(kernel)

        @contextlib.contextmanager
        def ctx():
            with kernel:
                yield kernel

        return ctx()

    def iter_runable_groups(self):
        while self.runable_reduction_groups or self.runable_pointwise_groups:
            if self.runable_reduction_groups:
                yield next(iter(self.runable_reduction_groups.keys()))
            else:
                yield next(iter(self.runable_pointwise_groups.keys()))
        assert not self.runable_nodes
        assert len(self.nodes) == self.run_count

    def iter_fixed_point(self):
        """
        Keep yielding until self.run_count converges
        """
        prior_run_count = -1
        while prior_run_count != self.run_count:
            prior_run_count = self.run_count
            yield

    def pop_group(self, group):
        while group in self.runable_nodes:
            self.runable_reduction_groups.pop(group, None)
            self.runable_pointwise_groups.pop(group, None)
            yield from self.runable_nodes.pop(group)
        if self.fusable_deps:
            fusable = True
            while fusable:
                fusable = self.blocked_nodes.pop_fusable(self.fusable_deps, group)
                yield from fusable

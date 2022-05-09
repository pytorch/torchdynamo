import collections
import contextlib
import dataclasses
import functools
from itertools import chain
from typing import Any
from typing import Dict
from typing import List

import numpy
import numpy as np

from . import config
from . import dependencies
from . import ir
from .dependencies import StarDep
from .virtualized import V


def cmp(a, b):
    return int(a > b) - int(a < b)


class OutputNode:
    def __init__(self, dep):
        self.unmet_dependencies = {dep}
        self.inverse_users = []

    def is_reduction(self):
        return False

    def get_alias_names(self):
        return ()


class BaseSchedulerNode:
    def __init__(self, scheduler: "Scheduler", node: ir.Buffer):
        self.scheduler = scheduler
        self.node = node
        self.users = None
        self.inverse_users = []
        self.set_read_writes(node.get_read_writes())

    def set_users(self, users: List["NodeUser"]):
        # deduplicate
        result = dict()
        for use in users:
            if id(use.node) in result:
                result[id(use.node)] = NodeUser(
                    use.node, result[id(use.node)].can_inplace and use.can_inplace
                )
            else:
                result[id(use.node)] = use
        self.users = list(result.values())

    def get_aliases(self):
        return self.node.get_alias_names()

    def set_read_writes(self, rw):
        self.read_writes = rw
        self.unmet_dependencies = self.read_writes.reads
        self.prune_deps()

    def prune_deps(self):
        self.unmet_dependencies = {
            dep
            for dep in self.unmet_dependencies
            if dep.name not in self.scheduler.available_buffer_names
        }

    def get_name(self):
        return self.node.get_name()

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        return False

    def allocate(self):
        if self.node.should_allocate():
            V.graph.wrapper_code.codegen_allocation(self.node)

    def can_free(self):
        for use in self.users:
            if isinstance(use.node, OutputNode):
                return False
            name = use.get_name()
            if name not in self.scheduler.available_buffer_names:
                return False
        return True


class ExternKernelSchdulerNode(BaseSchedulerNode):
    def can_remove_buffer(self, **kwargs):
        return False

    def run(self, codegen_extern_call):
        self.allocate()
        self.scheduler.run_count += 1
        self.scheduler.pending_buffer_names.add(self.get_name())
        self.scheduler.kernels.append(self.node)
        codegen_extern_call(self.node)


class NopKernelSchdulerNode(BaseSchedulerNode):
    def can_remove_buffer(self, **kwargs):
        return False

    def run(self):
        self.allocate()
        self.scheduler.run_count += 1
        self.scheduler.pending_buffer_names.add(self.get_name())


def pick_loop_order(stride_lengths, sizes):
    """
    A heuristic to decide loop iteration orders.  This has not been well
    tuned and may be something we should autotune.
    """

    @functools.cmp_to_key
    def index_cmp(a, b):
        if sizes[a] == 1 or sizes[b] == 1:
            # 1-sizes don't matter, just move them to the end
            return cmp(sizes[a] == 1, sizes[b] == 1)

        a_first = np.logical_or(
            stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]
        ).all()
        b_first = np.logical_or(
            stride_lengths[:, a] == 0, stride_lengths[:, a] > stride_lengths[:, b]
        ).all()

        if a_first and not b_first:
            return -1
        if b_first and not a_first:
            return 1

        # otherwise contiguous
        return cmp(b, a)

    order = list(reversed(range(stride_lengths.shape[1])))
    if config.pick_loop_orders:
        order.sort(key=index_cmp)
    return order


def inverse_reorder(order):
    inv_order = dict(zip(order, range(len(order))))

    def reindex(index):
        assert len(index) == len(inv_order)
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def apply_loop_reordering(stides, sizes):
    order = list(reversed(pick_loop_order(stides, sizes)))
    sizes = [sizes[i] for i in order]
    reindex2 = inverse_reorder(order)
    sizes, reindex1 = ir.SqueezeView.squeezer(sizes)

    def reindex(index):
        return reindex2(reindex1(index))

    return sizes, reindex


class SchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler: "Scheduler", node: ir.ComputedBuffer, group_fn):
        super().__init__(scheduler, node)

        _, (index_vars, reduction_vars) = dependencies.index_vars(
            node.get_size(), node.get_reduction_size()
        )
        rw = node.get_read_writes()
        memory_addrs = [dep.index for dep in chain(rw.reads, rw.writes)]
        stride_lengths = numpy.array(
            [
                V.graph.sizevars.stride_hints(expr, [*index_vars, *reduction_vars])
                for expr in memory_addrs
            ],
            dtype=numpy.int64,
        )

        index_strides = stride_lengths[:, : len(index_vars)]
        reduction_strides = stride_lengths[:, len(index_vars) :]
        assert index_strides.shape == (len(memory_addrs), len(index_vars))
        assert reduction_strides.shape == (len(memory_addrs), len(reduction_vars))

        self._size, self._reindex = apply_loop_reordering(
            index_strides, node.get_size()
        )
        self._reduction_size, self._reduction_reindex = apply_loop_reordering(
            reduction_strides, node.get_reduction_size()
        )

        self.group = (
            group_fn(self._size),
            group_fn(self._reduction_size),
        )

        # need to recompute reads/writes with possible loop reordering
        if node.get_reduction_type():

            def store_fn(vars, reduction_vars):
                return node.get_store_function()(
                    self._reindex(vars), self._reduction_reindex(reduction_vars)
                )

            self.set_read_writes(
                dependencies.extract_read_writes(
                    store_fn,
                    self._size,
                    self._reduction_size,
                )
            )
        else:

            def store_fn(vars):
                return node.get_store_function()(self._reindex(vars))

            self.set_read_writes(
                dependencies.extract_read_writes(
                    store_fn,
                    self._size,
                )
            )

    def can_remove_buffer(self, broadcast_after_reduce=False):
        if (
            self.is_reduction()
            and len(self.users) == 1
            and len(self.users[0].node.unmet_dependencies) == 1
        ):
            user = self.users[0].node
            dep = next(iter(user.unmet_dependencies))
            writes = self.read_writes.writes
            if broadcast_after_reduce:
                writes = set(writes)
                writes.update(
                    [w.broadcast_extend_sizes(self._reduction_size) for w in writes]
                )
            # this will get fused into us, so we don't need to keep the buffer
            return not user.is_reduction() and dep in writes
        return False

    def mark_fusable(self, broadcast_after_reduce=False):
        self.scheduler.fusable_deps.update(self.read_writes.writes)
        if broadcast_after_reduce and self._reduction_size:
            self.scheduler.fusable_deps.update(
                w.broadcast_extend_sizes(self._reduction_size)
                for w in self.read_writes.writes
            )

    def get_ranges(self):
        return self._size, self._reduction_size

    def is_reduction(self):
        return bool(self.node.data.get_reduction_type())

    def run(self, vars, reduction_vars):
        V.graph.wrapper_code.codegen_allocation(self.node)
        self.scheduler.run_count += 1
        name = self.get_name()
        indexer = self.node.layout.make_indexer()
        if self.is_reduction():
            vars = self._reindex(vars)
            reduction_vars = self._reduction_reindex(reduction_vars)
            self.node.data.store_reduction(name, indexer, vars, reduction_vars)
        else:
            vars = self._reindex([*vars, *reduction_vars])
            self.node.data.store_output(name, indexer, vars)
        self.scheduler.pending_buffer_names.add(self.get_name())

    def can_inplace(self, read_dep: dependencies.MemoryDep):
        if len(self.read_writes.writes) == 1 and hasattr(read_dep, "index"):
            write_dep = next(iter(self.read_writes.writes))
            return read_dep.index == write_dep.index and read_dep.size == write_dep.size
        return False


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


@dataclasses.dataclass
class NodeUser:
    node: BaseSchedulerNode
    can_inplace: bool = False

    def get_name(self):
        return self.node.get_name()


class Scheduler:
    def __init__(self, group_fn, nodes):
        super(Scheduler, self).__init__()
        self.group_fn = group_fn
        self.runable_reduction_groups = collections.defaultdict(int)
        self.runable_pointwise_groups = collections.defaultdict(int)
        self.runable_nodes: Dict[Any, SchedulerNode] = collections.defaultdict(list)
        self.runable_extern_kernels = collections.deque()
        self.blocked_nodes = BlockedNodes()
        self.run_count = 0
        self.nodes = []
        self.kernels = []
        self.available_buffer_names = {
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
        }
        self.pending_buffer_names = set()
        self.check_can_free = set()
        self.fusable_deps = set()
        for node in nodes:
            if node.is_no_op():
                self.nodes.append(NopKernelSchdulerNode(self, node))
            elif isinstance(node, ir.ComputedBuffer):
                self.nodes.append(SchedulerNode(self, node, self.group_fn))
            elif isinstance(node, ir.ExternKernel):
                self.nodes.append(ExternKernelSchdulerNode(self, node))
            else:
                assert False, node
        self.compute_users()
        self.name_to_node = {node.get_name(): node for node in self.nodes}
        self.enqueue(self.nodes)

    def compute_users(self):
        name_to_users = collections.defaultdict(list)
        for node in V.graph.graph_outputs:
            name_to_users[node.get_name()].append(
                NodeUser(OutputNode(StarDep(node.get_name())))
            )

        # handle aliasing
        for node1 in self.nodes:
            node1_name = node1.get_name()
            for node2_name in node1.get_aliases():
                if node1_name in name_to_users and node2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[node1_name]
                    list2 = name_to_users[node2_name]
                    combined = list1 + list2
                    for key in name_to_users.keys():
                        if name_to_users[key] is list1 or name_to_users[key] is list2:
                            name_to_users[key] = combined
                elif node1_name in name_to_users:
                    name_to_users[node2_name] = name_to_users[node1_name]
                else:
                    name_to_users[node1_name] = name_to_users[node2_name]

        for node in reversed(self.nodes):
            node.set_users(name_to_users[node.get_name()])
            name_to_users[node.get_name()] = None
            for read in node.read_writes.reads:
                name_to_users[read.name].append(NodeUser(node, node.can_inplace(read)))

        for node in self.nodes:
            for user in node.users:
                user.node.inverse_users.append(node)

        updated_nodes = []
        for node in self.nodes:
            if node.users:
                updated_nodes.append(node)
            else:
                # dead code
                V.graph.removed_buffers.add(node.get_name())
        self.nodes = updated_nodes

    def maybe_remove_buffer(self, node: SchedulerNode, broadcast_after_reduce=False):
        if node.can_remove_buffer(broadcast_after_reduce=broadcast_after_reduce):
            V.graph.removed_buffers.add(node.get_name())

    def enqueue(self, node):
        if isinstance(node, (tuple, list)):
            for n in node:
                self.enqueue(n)
            return

        assert isinstance(node, BaseSchedulerNode)
        if node.unmet_dependencies:
            self.blocked_nodes.add(node)
        else:
            if isinstance(node, ExternKernelSchdulerNode):
                self.runable_extern_kernels.append(node)
            elif isinstance(node, NopKernelSchdulerNode):
                node.run()
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
        while self.pending_buffer_names:
            self.available_buffer_names.update(self.pending_buffer_names)
            nodes_to_add = []
            for name in self.pending_buffer_names:
                self.check_can_free.update(self.pending_buffer_names)
                for node in self.blocked_nodes.pop_name(name):
                    node.prune_deps()
                    nodes_to_add.append(node)
            self.pending_buffer_names.clear()
            self.enqueue(nodes_to_add)

    def maybe_free_buffers(self):
        # perhaps there are some unused buffers we can free
        for done_name in self.check_can_free:
            done_node = self.name_to_node[done_name]
            for node in done_node.inverse_users:
                if node.can_free():
                    V.graph.wrapper_code.codegen_free(node.node)
        self.check_can_free.clear()

    def kernel(self, kernel):
        self.fusable_deps.clear()
        self.kernels.append(kernel)

        @contextlib.contextmanager
        def ctx():
            with kernel:
                yield kernel

        return ctx()

    def iter_runable_groups(self, codegen_extern_call):
        while (
            self.runable_reduction_groups
            or self.runable_pointwise_groups
            or self.runable_extern_kernels
        ):
            if self.runable_extern_kernels:
                self.runable_extern_kernels.popleft().run(codegen_extern_call)
            elif self.runable_reduction_groups:
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

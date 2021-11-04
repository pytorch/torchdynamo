import dataclasses
import dis

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_ABSOLUTE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # dis.opmap["RERAISE"],
    # TODO(jansel): double check exception handling
}
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)


def remove_dead_code(instructions):
    """Dead code elimination"""
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    live_code = set()

    def find_live_code(start):
        for i in range(start, len(instructions)):
            if i in live_code:
                return
            live_code.add(i)
            inst = instructions[i]
            if inst.opcode in JUMP_OPCODES:
                find_live_code(indexof[id(inst.target)])
            if inst.opcode in TERMINAL_OPCODES:
                return

    find_live_code(0)
    return [inst for i, inst in enumerate(instructions) if i in live_code]


def remove_pointless_jumps(instructions):
    """Eliminate jumps to the next instruction"""
    pointless_jumps = {
        id(a)
        for a, b in zip(instructions, instructions[1:])
        if a.opname == "JUMP_ABSOLUTE" and a.target is b
    }
    return [inst for inst in instructions if id(inst) not in pointless_jumps]


@dataclasses.dataclass
class ReadsWrites:
    reads: set
    writes: set
    visited: set


def livevars_analysis(instructions, instruction):
    indexof = {id(inst): i for i, inst in enumerate(instructions)}
    must = ReadsWrites(set(), set(), set())
    may = ReadsWrites(set(), set(), set())

    def walk(state, start):
        if start in state.visited:
            return
        state.visited.add(start)

        for i in range(start, len(instructions)):
            inst = instructions[i]
            if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
                if "LOAD" in inst.opname:
                    if inst.argval not in must.writes:
                        state.reads.add(inst.argval)
                elif "STORE" in inst.opname or "DELETE" in inst.opname:
                    state.writes.add(inst.argval)
                else:
                    assert False, f"unhandled {inst.opname}"
            if inst.opcode in JUMP_OPCODES:
                walk(may, indexof[id(inst.target)])
                state = may
            if inst.opcode in TERMINAL_OPCODES:
                return

    walk(must, indexof[id(instruction)])
    return must.reads | may.reads

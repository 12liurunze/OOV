"""Fast, exact equality for scheduler-owned native integer arrays."""
from array import array
from typing import NamedTuple

from .repair_lifecycle import PrefixEntry, PrefixLedger, StaleAllocation


class Snapshot(NamedTuple):
    entry: PrefixEntry
    prompt_tuple: tuple
    committed_tuple: tuple
    prompt_array: array
    committed_array: array


class ArrayPrefixLedger(PrefixLedger):
    def __init__(self):
        super().__init__()
        self._snapshots = {}

    def observe(self, slot, expected, live, owner, prompt, outputs, prefix_count):
        arrays = (
            type(prompt) is array and prompt.typecode == "q"
            and type(outputs) is array and outputs.typecode == "q"
        )
        if type(slot) is not int:
            return super().observe(slot, expected, live, owner, prompt, outputs, prefix_count)
        snap = self._snapshots.get(slot)
        if (
            arrays and snap is not None and slot > 0
            and type(expected) is tuple and type(live) is tuple
            and len(expected) > 1 and expected == live and expected[1] == slot
            and snap.entry is self.entries.get(slot)
            and snap.entry.identity == expected and snap.entry.owner is owner
            and snap.entry.prompt is snap.prompt_tuple
            and snap.entry.committed is snap.committed_tuple
        ):
            if prompt != snap.prompt_array:
                raise StaleAllocation("prompt changed within the same allocation")
            end = min(max(0, int(prefix_count)), len(outputs))
            old = len(snap.committed_tuple)
            if end < old:
                return None
            if outputs[:old] != snap.committed_array:
                raise StaleAllocation("already committed output was rewritten")
            if end > old:
                delta = outputs[old:end]
                snap.entry.committed += tuple(delta)
                snap.committed_array.extend(delta)
                self._snapshots[slot] = Snapshot(
                    snap.entry, snap.prompt_tuple, snap.entry.committed,
                    snap.prompt_array, snap.committed_array,
                )
            return snap.entry

        # All allocation transitions and unexpected container types use the
        # original implementation. A tuple could retain an equal custom value
        # from an earlier non-array prompt, so cache only original integer tuples.
        self._snapshots.pop(slot, None)
        entry = super().observe(slot, expected, live, owner, prompt, outputs, prefix_count)
        if arrays and entry is not None and all(type(token) is int for token in entry.prompt):
            self._snapshots[slot] = Snapshot(
                entry, entry.prompt, entry.committed, prompt[:], outputs[:len(entry.committed)],
            )
        return entry

    def clear(self):
        super().clear()
        self._snapshots.clear()

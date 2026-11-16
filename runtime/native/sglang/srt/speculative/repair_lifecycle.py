"""CPU-only request identity and committed-prefix state, shared with unit tests."""
from dataclasses import dataclass


class StaleAllocation(RuntimeError):
    pass


@dataclass
class PrefixEntry:
    identity: tuple
    owner: object
    prompt: tuple
    committed: tuple = ()


class PrefixLedger:
    def __init__(self):
        self.entries = {}

    def observe(self, slot, expected, live, owner, prompt, outputs, prefix_count):
        """None means an older overlapping prefix: do not read or mutate its pool.

        A different rid is NOT the allocation identity. Generation and reset epoch
        prevent reuse even when the caller reuses its request id or Req instance.
        Holding the actual Req reference also avoids Python object-id ABA.
        """
        if slot <= 0 or expected is None or tuple(expected) != tuple(live):
            raise StaleAllocation('request allocation is stale, absent, or padding')
        if tuple(expected)[1] != slot:
            raise StaleAllocation('batch slot differs from allocation identity')
        prompt = tuple(prompt)
        end = min(max(0, int(prefix_count)), len(outputs))
        entry = self.entries.get(slot)
        identity = tuple(expected)
        if entry is None or entry.identity != identity or entry.owner is not owner:
            entry = PrefixEntry(identity, owner, prompt)
            self.entries[slot] = entry
        elif entry.prompt != prompt:
            raise StaleAllocation('prompt changed within the same allocation')
        old = len(entry.committed)
        if end < old:
            return None
        committed = tuple(outputs[:end])
        if committed[:old] != entry.committed:
            raise StaleAllocation('already committed output was rewritten')
        entry.committed = committed
        return entry

    def clear(self):
        self.entries.clear()


def validate_execution_scope(*, eagle3, dynamic_budget, dp_size, pp_size,
                             enable_dp_attention, enable_lora, lora_paths):
    """Reject modes that invalidate this frozen scorer/tree's tested contract."""
    if not eagle3:
        raise ValueError('ContextRepair requires EAGLE3')
    if dynamic_budget:
        raise ValueError('ContextRepair does not support runtime speculative budget changes')
    if dp_size != 1 or pp_size != 1 or enable_dp_attention:
        raise ValueError('ContextRepair requires DP1 and PP1 without DP attention')
    if enable_lora or lora_paths:
        raise ValueError('ContextRepair frozen model identity does not support LoRA')


def readout_shard_bounds(head, group, label):
    """Check the contiguous, original-token rows owned by this TP rank."""
    world = group.world_size
    rank = group.rank_in_group
    if world < 1 or not 0 <= rank < world or head.tp_size != world:
        raise ValueError(f'{label} vocabulary head does not match the TP group')
    rows, remainder = divmod(head.org_vocab_size_padded, world)
    if remainder or rows != head.weight.shape[0]:
        raise ValueError(f'{label} vocabulary head has an incompatible shard size')
    low = min(rank * rows, head.org_vocab_size)
    high = min((rank + 1) * rows, head.org_vocab_size)
    shard = head.shard_indices
    if (shard.org_vocab_start_index, shard.org_vocab_end_index) != (low, high):
        raise ValueError(f'{label} vocabulary head does not own the expected token IDs')
    return low, high


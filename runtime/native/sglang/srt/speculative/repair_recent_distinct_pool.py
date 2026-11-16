"""Incremental recent-distinct membership on the allocation-checked prefix."""

from array import array
from itertools import islice
from typing import NamedTuple

from .repair_calibration.candidates import CandidatePool


class State(NamedTuple):
    entry: object
    length: int
    recent: dict
    pool: CandidatePool


class IncrementalRecentDistinctPool:
    def __init__(self, generator):
        if generator.policy != "recent_distinct_v1":
            raise ValueError("Expected the frozen recent-distinct selector")
        self.generator = generator
        self.hot = generator.hot_vocabulary.ids
        self.vocab = generator.target_vocab_size
        self.capacity = generator.capacity
        self.states = {}
        self.rebuilds = self.delta_updates = self.reused = self.fallbacks = 0

    def clear(self):
        self.states.clear()

    def receipt(self):
        return dict(rebuilds=self.rebuilds, delta_updates=self.delta_updates,
                    reused=self.reused, fallbacks=self.fallbacks)

    def _cold_tokens(self, tokens):
        cold = []
        for token in tokens:
            if type(token) is not int or not 0 <= token < self.vocab:
                raise ValueError("invalid target token id")
            if token not in self.hot:
                cold.append(token)
        return cold

    def observe(self, slot, entry, prompt, outputs):
        # The ledger already checked ownership and exact prefix *values*.
        # Require integer arrays since Python equality also treats e.g. 1.0 as 1.
        if (type(prompt) is not array or prompt.typecode != "q"
                or type(outputs) is not array or outputs.typecode != "q"):
            self.states.pop(slot, None)
            self.fallbacks += 1
            return self.generator(entry.prompt, entry.committed)

        end = len(entry.committed)
        previous = self.states.get(slot)
        if previous is None or previous.entry is not entry:
            added = self._cold_tokens(entry.prompt) + self._cold_tokens(entry.committed)
            recent = {}
            self.rebuilds += 1
        else:
            if end < previous.length:
                raise RuntimeError("Recent-distinct state is ahead of the verified prefix")
            if end == previous.length:
                self.reused += 1
                return previous.pool
            # Validate the entire delta before mutating state.
            added = self._cold_tokens(entry.committed[previous.length:])
            recent = previous.recent
            if not added:
                self.reused += 1
                self.states[slot] = State(entry, end, recent, previous.pool)
                return previous.pool
            self.delta_updates += 1

        for token in added:
            recent.pop(token, None)
            recent[token] = None
        count = len(recent)
        ids = tuple(sorted(islice(reversed(recent), self.capacity)))
        pool = CandidatePool(ids, None, ids, bool(ids), count > self.capacity,
                             count, self.generator.policy)
        self.states[slot] = State(entry, end, recent, pool)
        return pool

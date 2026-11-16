"""Reuse formatted stop rows for each verified request allocation."""
from typing import NamedTuple


EMPTY_SET = frozenset()
EMPTY_ROW = (-1,) * 32
UNSUPPORTED = object()


class StopEntry(NamedTuple):
    owner: object
    request_stops: frozenset
    model_stops: frozenset
    tokenizer_stops: tuple
    result: tuple


class StopMetadataCache:
    def __init__(self, vocab):
        self.vocab = vocab
        self.entries = {}
        self.hits = 0
        self.misses = 0
        self.inactive = 0
        self.fallbacks = 0

    def clear(self):
        self.entries.clear()

    @staticmethod
    def tokenizer_stops(tokenizer, batch_tokenizers):
        if tokenizer is None:
            return None, EMPTY_SET
        kind = type(tokenizer)
        if (not kind.__module__.startswith("transformers.")
                or kind.__name__ not in ("TokenizersBackend", "PreTrainedTokenizerFast")):
            return UNSUPPORTED
        key = id(tokenizer)
        if key not in batch_tokenizers:
            # The scheduler loads/attaches one tokenizer and runs pure decode
            # synchronously. Refresh the resolved IDs once per batch, including
            # in-place additions to the SGLang-owned additional-stop set.
            eos = tokenizer.eos_token_id
            extra = getattr(tokenizer, "additional_stop_token_ids", None)
            if extra is not None and type(extra) not in (set, frozenset):
                batch_tokenizers[key] = UNSUPPORTED
            else:
                batch_tokenizers[key] = eos, frozenset(extra) if extra else EMPTY_SET
        return batch_tokenizers[key]

    def row(self, slot, entry, req, params, batch_tokenizers):
        # Admission/clipping may change sampling scalars. Read the actual
        # parameters and the ledger's committed length at every call.
        if params.ignore_eos or len(entry.committed) < (params.min_new_tokens or 0):
            self.inactive += 1
            return EMPTY_ROW, False
        request_stops = params.stop_token_ids
        model_stops = req.eos_token_ids
        if any(value is not None and type(value) not in (set, frozenset)
               for value in (request_stops, model_stops)):
            self.fallbacks += 1
            return None
        tokenizer_stops = self.tokenizer_stops(req.tokenizer, batch_tokenizers)
        if tokenizer_stops is UNSUPPORTED:
            self.fallbacks += 1
            return None
        request_stops = request_stops or EMPTY_SET
        model_stops = model_stops or EMPTY_SET
        old = self.entries.get(slot)
        if (old is not None and old.owner is entry and old.request_stops == request_stops
                and old.model_stops == model_stops and old.tokenizer_stops == tokenizer_stops):
            self.hits += 1
            return old.result

        eos, extra = tokenizer_stops
        combined = set(request_stops)
        combined.update(model_stops)
        if eos is not None:
            combined.add(eos)
        combined.update(extra)
        # The original implementation checks the raw unique count before
        # discarding out-of-vocabulary IDs.
        deferred = len(combined) > 32
        if deferred:
            result = EMPTY_ROW, True
        else:
            ordered = tuple(sorted(token for token in combined if 0 <= token < self.vocab))
            result = ordered + (-1,) * (32 - len(ordered)), False
        self.entries[slot] = StopEntry(entry, frozenset(request_stops), frozenset(model_stops),
                                       tokenizer_stops, result)
        self.misses += 1
        return result

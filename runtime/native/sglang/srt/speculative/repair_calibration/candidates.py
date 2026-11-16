"""The single, pure-CPU candidate generator for collection and request updates.

Only confirmed tokens may be supplied. A caller must enforce the request owner,
generation and asynchronous prefix watermark before calling this function.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

POLICIES = ("legacy_all_cold_or_disable_v1", "recent_distinct_v1")
ROOT_POLICY = "separate_cold_root_deduplicate_v1"
TIE_POLICY = "largest_finite_score_then_smallest_target_id_v1"


@dataclass(frozen=True)
class CandidatePool:
    pool_ids: tuple
    root_token: object
    candidate_ids: tuple
    available: bool
    overflow: bool
    all_cold_count: int
    policy: str


@dataclass(frozen=True)
class FrozenHotVocabulary:
    """Validate static draft-to-target membership once, then reuse immutably."""
    ids: frozenset
    target_vocab_size: int = 128256

    def __post_init__(self):
        if type(self.target_vocab_size) is not int or self.target_vocab_size < 1:
            raise ValueError("target vocabulary size must be a positive integer")
        frozen = frozenset(self.ids)
        if any(type(t) is not int or not 0 <= t < self.target_vocab_size for t in frozen):
            raise ValueError("invalid hot-map target id")
        object.__setattr__(self, "ids", frozen)

    def __contains__(self, token):
        return token in self.ids

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)


@dataclass(frozen=True, init=False)
class CandidateGenerator:
    """Per-worker generator: calls scan visible tokens, never the hot vocabulary.

    Both static policy and vocabulary are validated at construction. This does
    not cache request state or waive the caller's confirmed-prefix/owner checks.
    """
    hot_vocabulary: FrozenHotVocabulary
    capacity: int
    policy: str

    def __init__(self, hot_ids, *, capacity=64, policy=POLICIES[0], target_vocab_size=None):
        if policy not in POLICIES:
            raise ValueError("unknown candidate policy")
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be positive integer")
        if isinstance(hot_ids, FrozenHotVocabulary):
            hot = hot_ids
            if target_vocab_size is not None and target_vocab_size != hot.target_vocab_size:
                raise ValueError("frozen hot vocabulary has a different target vocabulary size")
        else:
            hot = FrozenHotVocabulary(hot_ids, 128256 if target_vocab_size is None else target_vocab_size)
        object.__setattr__(self, "hot_vocabulary", hot)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "policy", policy)

    @property
    def target_vocab_size(self):
        return self.hot_vocabulary.target_vocab_size

    def __call__(self, prompt_ids, committed_output_ids, *, root_token=None):
        hot = self.hot_vocabulary.ids
        visible = list(prompt_ids) + list(committed_output_ids)
        for token in visible + ([] if root_token is None else [root_token]):
            if type(token) is not int or not 0 <= token < self.target_vocab_size:
                raise ValueError("invalid target token id")
        last = {t: i for i, t in enumerate(visible) if t not in hot}
        overflow = len(last) > self.capacity
        if overflow and self.policy == POLICIES[0]:
            return CandidatePool((), None, (), False, True, len(last), self.policy)
        retained = (sorted(last, key=last.get)[-self.capacity:]
                    if self.policy == POLICIES[1] else list(last))
        pool = tuple(sorted(retained))
        root = root_token if root_token is not None and root_token not in hot else None
        candidates = tuple(sorted(set(pool) | ({root} if root is not None else set())))
        return CandidatePool(pool, root, candidates, bool(candidates), overflow,
                             len(last), self.policy)


def generator_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def generate_candidates(prompt_ids, committed_output_ids, hot_ids, *,
                        root_token=None, capacity=64,
                        policy=POLICIES[0], target_vocab_size=128256):
    """Compatibility wrapper. Hot paths should reuse CandidateGenerator instead."""
    generator = CandidateGenerator(hot_ids, capacity=capacity, policy=policy,
                                   target_vocab_size=target_vocab_size)
    return generator(prompt_ids, committed_output_ids, root_token=root_token)


def select_scored_candidate(candidate_ids, scores):
    """Nonfinite scores are not candidates; no finite score means no X."""
    import math
    if len(candidate_ids) != len(scores):
        raise ValueError("candidate and score lengths differ")
    finite = [(float(s), int(t)) for t, s in zip(candidate_ids, scores)
              if math.isfinite(float(s))]
    if not finite:
        return None, None
    score, token = min(finite, key=lambda v: (-v[0], v[1]))
    return token, score

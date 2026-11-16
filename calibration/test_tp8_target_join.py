"""CPU-only tests for exact-prefix Target receipts and calibration trace joining."""

import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from .candidates import ROOT_POLICY, TIE_POLICY, generator_sha256
from .contracts import (CANDIDATE_PAIR_POLICY, DRAFT_KV_ANCESTRY, FEATURE_NAMES,
                        FEATURE_SCHEMA, ROW_SCHEMA, hot_map_digest, prefix_hash)
from .tp8_join import join_capture
from .tp8_target import exact_committed_prefix, tree_paths, validate_mask_row, verify_rows


def make_contract(hot):
    digest = "a" * 64
    return {
        "target_model_id": "/target", "target_weights_sha256": digest,
        "draft_model_id": "/draft", "draft_weights_sha256": digest,
        "tokenizer_sha256": digest, "alignment_sha256": digest,
        "hot_map_sha256": hot_map_digest(hot), "candidate_generator_sha256": generator_sha256(),
        "candidate_policy": "recent_distinct_v1", "capacity": 64, "root_policy": ROOT_POLICY,
        "tie_policy": TIE_POLICY, "readout_backend": "fused_cached_v1",
        "readout_numeric_version": "cached_WA_bf16_v1", "hidden_size": 8192,
        "target_vocab_size": 128256, "draft_vocab_size": len(hot),
        "dtype": "bfloat16", "tp_size": 8, "max_steps": 7, "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES), "label_schema": ROW_SCHEMA, "sampling_mode": "greedy",
        "feature_dim": len(FEATURE_NAMES), "score_mode": "unified_target_event_v2",
        "candidate_pair_policy": CANDIDATE_PAIR_POLICY, "draft_kv_ancestry": DRAFT_KV_ANCESTRY,
    }


class TargetTreeTest(unittest.TestCase):
    def test_prefix_commit_stops_before_bonus_root(self):
        request = SimpleNamespace(rid="cal", origin_input_ids=[10, 11], output_ids=[20, 21],
                                  multimodal_inputs=None, is_retracted=False, finished=lambda: False)
        self.assertEqual(exact_committed_prefix(request, 3, 21), [10, 11, 20])
        with self.assertRaisesRegex(RuntimeError, "prefix/root mismatch"):
            exact_committed_prefix(request, 4, 21)

    def test_mask_detects_sibling_leakage(self):
        paths = tree_paths([1, 3, -1, -1], [-1, 2, -1, -1])
        self.assertEqual(paths[3], (0, 1, 3))
        validate_mask_row([1, 1, 1, 1, 0, 1], 2, paths, 3)
        with self.assertRaisesRegex(ValueError, "tree path"):
            validate_mask_row([1, 1, 1, 1, 1, 1], 2, paths, 3)
        with self.assertRaisesRegex(ValueError, "cycle"):
            tree_paths([1, 0], [-1, -1])

    def test_verify_rows_uses_actual_logits_and_request_major_offsets(self):
        import torch

        lengths = (2, 3)
        requests = [SimpleNamespace(
            rid=str(i), origin_input_ids=list(range(length)), output_ids=[11 + i],
            multimodal_inputs=None, is_retracted=False, finished=lambda: False,
        ) for i, length in enumerate(lengths)]
        batch = SimpleNamespace(reqs=requests, seq_lens=torch.tensor(lengths),
                                enable_overlap=False, has_grammar=False,
                                sampling_info=SimpleNamespace(is_all_greedy=True))
        masks, tokens, positions = [], [], []
        for request_index, length in enumerate(lengths):
            for n in range(8):
                masks.extend([1] * length + [1] * (n + 1) + [0] * (7 - n))
                tokens.append(11 + request_index if n == 0 else 16 + n)
                positions.append(length + n)
        verify = SimpleNamespace(
            draft_token_num=8, draft_token=torch.tensor(tokens),
            positions=torch.tensor(positions), custom_mask=torch.tensor(masks, dtype=torch.uint8),
            retrieve_index=torch.arange(16), retrieve_next_sibling=torch.full((2, 8), -1),
            retrieve_next_token=torch.tensor([[1, 2, 3, 4, 5, 6, 7, -1]] * 2),
        )
        logits = torch.zeros(16, 128256)
        logits[torch.arange(16), torch.arange(16) + 100] = 1
        model_output = SimpleNamespace(next_token_logits=logits)
        rows = list(verify_rows(verify, batch, model_output, 9,
                                graph_selected=False, grammar_mask=None))
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[8]["parent_prefix_ids"], [0, 1, 2, 12])
        self.assertEqual(rows[8]["gold"], 108)
        self.assertEqual(rows[8]["target_execution_id"], "verify-9-request-1-node-0")
        verify.custom_mask[8 * (2 + 8) + 3 + 1] = 1
        with self.assertRaisesRegex(ValueError, "tree path"):
            list(verify_rows(verify, batch, model_output, 9,
                             graph_selected=False, grammar_mask=None))


class JoinTraceTest(unittest.TestCase):
    def test_join_keeps_real_x_parent_and_excludes_pruned_and_ambiguous_rows(self):
        hot = list(range(32000))
        x = 32007
        prompt = [42, x]
        root = 23
        committed = [55]
        visible = prompt + committed
        visible_sha = prefix_hash(visible)
        anchor = dict(request_id="cal70~000001", visible_prefix_sha256=visible_sha,
                      root_token=root, round_id=f"{visible_sha}-{root}")

        def draft(ancestors, kind, parity=True):
            return dict(anchor, parent_prefix_ids=visible + [root] + ancestors,
                        parent_kind=kind, depth=len(ancestors), prompt_ids=prompt,
                        committed_output_ids=committed, candidate_ids=[x],
                        candidate_scores=[0.12] if parity else [None], x_token=x, has_x=True,
                        native1=45, native2=46, p1=0.5, p2=0.2, native2_logit=-0.34,
                        native_parity=True, readout_parity=parity,
                        probe_replaced_b=not ancestors)

        def target(row, gold, execution):
            prefix = row["parent_prefix_ids"]
            return dict(anchor, parent_prefix_ids=prefix, position=len(prefix)-1,
                        target_prefix_sha256=prefix_hash(prefix), gold=gold,
                        target_execution_id=execution, capture_before_sampling=True, mask_parity=True)

        root_row, x_row = draft([], "root"), draft([x], "x")
        pruned = draft([57], "native2")
        ambiguous = draft([45], "native1")
        bad_parity = draft([46], "native2", parity=False)
        with tempfile.TemporaryDirectory() as folder:
            root_path = Path(folder)
            inputs = {"draft": [root_row, x_row, pruned, ambiguous, bad_parity],
                      "target": [target(root_row, x, "v0"), target(x_row, 46, "v1"),
                                 target(ambiguous, 1, "v2"), target(ambiguous, 2, "v3"),
                                 target(bad_parity, 46, "v4")],
                      "prompts": [dict(request_id=anchor["request_id"], canonical_question_id="q1",
                                       source="synthetic", question_content_sha256=prefix_hash(prompt),
                                       prompt_ids=prompt)]}
            for name, rows in inputs.items():
                (root_path / f"{name}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            (root_path / "contract.json").write_text(json.dumps(make_contract(hot)), encoding="utf-8")
            (root_path / "hot.json").write_text(json.dumps(hot), encoding="utf-8")
            report = join_capture(
                draft_path=root_path / "draft.jsonl", target_path=root_path / "target.jsonl",
                prompts_path=root_path / "prompts.jsonl", contract_path=root_path / "contract.json",
                hot_map_path=root_path / "hot.json", output_dir=root_path)
            self.assertEqual(report["kept_parent_kinds"], {"root": 1, "x": 1})
            self.assertEqual(report["kept_target_events"], {"x": 1, "native2": 1})
            self.assertEqual(report["excluded"], {
                "target_labels_disagree_for_same_prefix": 1, "not_in_actual_verify_tree": 1,
                "draft_parity_unavailable_or_failed": 1})
            self.assertEqual(report["counts"]["probe_replaced_b"], 1)
            with gzip.open(root_path / "exact_rows.jsonl.gz", "rt", encoding="utf-8") as stream:
                records = list(map(json.loads, stream))
            self.assertEqual([record["gold"] for record in records], [x, 46])
            self.assertEqual(records[1]["target_prefix_sha256"],
                             prefix_hash(visible + [root, x]))


if __name__ == "__main__":
    unittest.main()

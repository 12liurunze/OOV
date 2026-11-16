"""Fit the 70B vocabulary alignment from the two local output heads."""

import argparse
import json
from pathlib import Path

from .contracts import hot_map_digest, sha256_file


TARGET_VOCAB = 128256
DRAFT_VOCAB = 32000
HIDDEN = 8192
HEAD = "lm_head.weight"


def weight_file(directory, key):
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weights = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        matches = [name for name in (key, f"model.{key}") if name in weights]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {key} in {index}")
        name = matches[0]
        path = (directory / weights[name]).resolve(strict=True)
        if directory not in path.parents or path.suffix != ".safetensors":
            raise ValueError("Weight index points outside the model directory")
        return path, name
    from safetensors import safe_open

    path = (directory / "model.safetensors").resolve(strict=True)
    if directory not in path.parents:
        raise ValueError("Use a local checkpoint with real safetensors files")
    with safe_open(path, framework="pt", device="cpu") as weights:
        matches = [name for name in (key, f"model.{key}") if name in weights.keys()]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {key} in {path}")
    return path, matches[0]


def read_tensor(directory, key):
    from safetensors import safe_open

    path, name = weight_file(directory, key)
    with safe_open(path, framework="pt", device="cpu") as weights:
        return weights.get_tensor(name)


def fit(target, draft, hot, torch):
    heldout = ((hot * 2654435761) % 4294967296) % 5 == 0
    train_count = int((~heldout).sum())
    holdout_count = int(heldout.sum())
    if holdout_count < 5000 or train_count < 24000:
        raise ValueError("Invalid vocabulary holdout sample sizes")
    tx, ty = target[~heldout], draft[~heldout]
    vx, vy = target[heldout], draft[heldout]
    gram, cross = tx.T @ tx, tx.T @ ty
    del tx, ty
    scale = gram.diagonal().mean()
    eye = torch.eye(target.shape[1], dtype=target.dtype, device=target.device)
    trials = []
    for alpha in (0.01, 0.1, 1.0):
        chol = torch.linalg.cholesky(gram + alpha * scale * eye)
        projection = torch.cholesky_solve(cross, chol)
        error = float((vx @ projection - vy).norm() / vy.norm())
        trials.append({"alpha": alpha, "heldout_relative_l2": error})
        del chol, projection
    alpha = min(trials, key=lambda trial: trial["heldout_relative_l2"])["alpha"]
    gram, cross = target.T @ target, target.T @ draft
    chol = torch.linalg.cholesky(gram + alpha * gram.diagonal().mean() * eye)
    transform = torch.cholesky_solve(cross, chol)
    if not bool(torch.isfinite(transform).all()):
        raise ValueError("Alignment contains non-finite values")
    return transform.cpu(), {
        "token_train": train_count,
        "token_validation": holdout_count,
        "trials": trials,
        "chosen_alpha": alpha,
        "orientation": "target_head @ transform approximates draft_head",
    }


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    target, draft = (p.resolve(strict=True) for p in (args.target, args.draft))
    out = args.out.resolve()
    if out.exists():
        parser.error("choose a new output directory")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        parser.error("--device must be cpu or cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; --device cpu is supported and slower")
    d2t = read_tensor(draft, "d2t")
    if d2t.shape != (DRAFT_VOCAB,) or d2t.dtype != torch.int64:
        parser.error("expected the SpecForge 32000-element int64 d2t vector")
    hot = torch.arange(DRAFT_VOCAB, dtype=torch.int64) + d2t
    if len(torch.unique(hot)) != DRAFT_VOCAB or int(hot.min()) < 0 or int(hot.max()) >= TARGET_VOCAB:
        parser.error("Draft-to-Target mapping is out of range or non-unique")
    t2d = read_tensor(draft, "t2d")
    expected = torch.zeros(TARGET_VOCAB, dtype=torch.bool)
    expected[hot] = True
    if t2d.dtype != torch.bool or not torch.equal(t2d, expected):
        parser.error("Draft t2d membership does not match the d2t mapping")
    target_head, draft_head = (read_tensor(path, HEAD) for path in (target, draft))
    if target_head.shape != (TARGET_VOCAB, HIDDEN) or draft_head.shape != (DRAFT_VOCAB, HIDDEN):
        parser.error("output heads do not have the expected 70B dimensions")
    if target_head.dtype != torch.bfloat16 or draft_head.dtype != torch.bfloat16:
        parser.error("requires the original BF16 checkpoints")
    x = target_head.index_select(0, hot).to(device, dtype=torch.float32)
    y = draft_head.to(device, dtype=torch.float32)
    del target_head, draft_head
    torch.set_float32_matmul_precision("highest")
    with torch.no_grad():
        transform, report = fit(x, y, hot.to(device), torch)
    out.mkdir(parents=True, exist_ok=False)
    torch.save({"transform": transform, "report": report}, out / "alignment.pt")
    hot_ids = hot.tolist()
    (out / "hot_map.json").write_text(json.dumps(hot_ids) + "\n", encoding="utf-8")
    report.update(
        schema="oov_weight_only_alignment_70b_v1",
        device=str(device),
        hot_map_sha256=hot_map_digest(hot_ids),
        target_vocab_size=TARGET_VOCAB,
        draft_vocab_size=DRAFT_VOCAB,
        hidden_size=HIDDEN,
        alignment_sha256=sha256_file(out / "alignment.pt"),
        classifier_trained=False,
    )
    (out / "alignment_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(out / "alignment_report.json")


if __name__ == "__main__":
    main()

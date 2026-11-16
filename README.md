# OOV repair for EAGLE-3

This repository contains the SGLang source overlays, the Red Hat input
preparation script, and the single-node A100 TP8 benchmark. It also includes
the experimental 70B OOV port, output-head alignment, and classifier fitting
code. Use [agent.md](agent.md) for benchmark setup and
[CALIBRATION_TP8.md](CALIBRATION_TP8.md) for collection and classifier training.

The calibration guide covers a 24-question smoke test and a 256-question
collection run. The repository ships the pinned row manifest and a rebuild
script so the exact prompts can be generated locally from their upstream
sources. See [the calibration data step](CALIBRATION_TP8.md#3-准备校准题并先跑小样本)
and [the Red Hat preparation step](agent.md#3-准备八个子集).

The 70B native EAGLE baseline is the first experiment. A measured 70B OOV
comparison requires genuine TP8 calibration rows, a fitted asset bound to
these checkpoints, and a real eight-GPU smoke test. Start with the readiness
table in the runbook.

Source provenance and the frozen upstream archive digest are recorded in
[runtime/manifest.json](runtime/manifest.json). The SGLang source is covered
by [LICENSE.sglang](LICENSE.sglang).

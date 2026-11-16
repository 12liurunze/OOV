# OOV repair for EAGLE-3

This repository contains the SGLang source overlays, the Red Hat input
preparation script, and the single-node A100 TP8 benchmark. It also includes
the experimental 70B OOV port, output-head alignment, and classifier fitting
code. The runbook is in [agent.md](agent.md).

The 70B native EAGLE baseline is the first experiment. A measured 70B OOV
comparison requires genuine TP8 calibration rows, a fitted asset bound to
these checkpoints, and a real eight-GPU smoke test. Start with the readiness
table in the runbook.

Source provenance and the frozen upstream archive digest are recorded in
[runtime/manifest.json](runtime/manifest.json). The SGLang source is covered
by [LICENSE.sglang](LICENSE.sglang).

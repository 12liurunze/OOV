# SGLang source notice

This release distributes research modifications to the SGLang Python runtime.
The SGLang code and its modifications are distributed under Apache License 2.0.
The imported source retains its original notices.

- [Upstream SGLang project](https://github.com/sgl-project/sglang)
- [Base snapshot provenance](https://github.com/12liurunze/llama-profile/blob/6d912fcc5ce55f2afcfa387aac504864ee33d0fc/vendor/provenance.json)
- [Apache License 2.0](LICENSE.sglang)

The frozen base archive was originally distributed with the Llama profile
research release. Its upstream Git commit is unavailable. The archive is pinned
by GitHub commit, file size, and SHA-256 in [the manifest](manifest.json).

All Python files listed under `arms.native.files` in the manifest are local
research additions or modifications relative to that base. The files under
`arms.repair.files` apply the additional OOV runtime changes used in the
experiment that disables the read-only request-slot version view. The installer
verifies the complete resulting source inventory for each arm.

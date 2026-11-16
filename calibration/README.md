# 70B TP8 补词校准

## 现有代码的运行边界

冻结的 Llama-3.1-8B 实验快照使用 4096 维。指定的两个 70B checkpoint 使用 8192 维。`tools/stage_70b.py` 会独立生成 8192 维 TP8 移植，保留跨 rank 归并和严格资产合同。`tools/collect_tp8.py` 启动独立的 eager 采集。真实 70B 的跨 rank 采集和正式 Graph 运行仍须在目标机器验收。 [完整操作步骤](../CALIBRATION_TP8.md)

这里提供两项独立可执行的校准步骤。第一项读取本机模型权重，生成 70B 词表对齐矩阵。第二项需要从独立的 70B TP8 采集得到真实的带标签前缀数据和匹配的合同，离线拟合分类器。权重、拟合出的参数和采集数据均保留在目标机器的本地目录。

## 生成词表对齐

在仓库根目录执行。三个路径均应在本机，建议输出到仓库之外。该步骤使用一张指定卡进行 FP32 岭回归。也可传 `--device cpu`。

```bash
python -m calibration.fit_alignment \
  --target "$TARGET_DIR" --draft "$DRAFT_DIR" \
  --out "$RUN_ROOT/alignment70" --device cuda:0
```

输出 `alignment.pt`、`hot_map.json` 和 `alignment_report.json`。脚本以实际 `d2t` 的有序映射配对输出头向量，用留出的词表行选择正则系数，然后用全部共享词表行重拟合。它核对 128256/32000 词表规模、8192 隐藏维度及 BF16 权重。输出只衡量输出头线性拟合误差，不提供在线命中率或性能。

## 用真实 70B TP8 已采集标签拟合分类器

按 [操作步骤](../CALIBRATION_TP8.md) 用同一 checkpoint、对齐矩阵、TP8 并行方式及补词候选策略收集样本。字段及协议见 [contracts.py](contracts.py) 的 `BOUND_FIELDS` 与 `validate_collection_row`。每行需要完整条件前缀匹配、真实 Target 标签和 Draft 特征证据。采集应覆盖 root、native1、native2、X 分支及保留 B 和保留 X 均有价值的训练样本。将实验评测题与训练题隔离，使用稳定的 canonical question ID。

```bash
python -m calibration.prepare_experiment \
  --rows /local/70b-tp8/exact_rows.jsonl.gz \
  --contract /local/70b-tp8/runtime_contract.json \
  --alignment-dir "$RUN_ROOT/alignment70" \
  --out "$RUN_ROOT/fitted70-tp8"
```

`fit/` 保存样本排除原因、各分支覆盖统计和训练/验证/测试指标。分别保留 `parameters/runtime_contract.json`、`parameters/alignment.pt`、`parameters/calibration_asset.json` 和 `receipt.json` 供后续复核。分类器选择保留第一个原生候选的实验策略。训练结果保留研究状态，后续按 [运行说明](../agent.md) 进行模型内验收和独立题目评估。

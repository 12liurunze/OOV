# 70B TP8 补词校准与分类器训练

本说明针对同一节点的 8 张 A100、Llama-3.3-70B-Instruct 和配套的 SpecForge EAGLE3。Target 和 Draft 权重保持冻结。采集的是模型运行时的候选和 Target 标签，训练的是替换第二候选的小型四分类器。

**首次使用需要在目标机通过短测。** 这个包包含采集和拟合代码。本地验证覆盖源码构建和 CPU 对照，目标机仍需确认真实 70B TP8 模型、跨卡归并及 Graph 路径。

## 代码从哪里开始读

| 环节 | 入口 | 工作内容 |
|---|---|---|
| 环境与任务 | [tools/collect_tp8.py](tools/collect_tp8.py) | 校验两套模型、对齐资产和八卡，构造独立的 eager 采集运行，启动一个 TP8 引擎 |
| 独立运行时 | [tools/stage_tp8_collection.py](tools/stage_tp8_collection.py) | 从冻结的 70B 运行时复制 `collect70b`，只在采集路径跳过分类器初始化。正式 benchmark 仍要求匹配的分类器资产 |
| Draft 侧 | [calibration/tp8_draft.py](calibration/tp8_draft.py) | 记录真实 Top2、词表外候选和分数。按固定种子在一部分可用状态上用 `X` 替代 `B`，保留 `A`。追踪分支实际如何展开 |
| Target 侧 | [calibration/tp8_target.py](calibration/tp8_target.py) | 观察 Target 验证中的真实树行，重建条件前缀，记录该条件下的 greedy Top1 |
| 合并与审核 | [calibration/tp8_join.py](calibration/tp8_join.py) | 按请求、轮次锚点和完整 token 前缀配对，验证有效记录和候选生成规则 |
| 拟合 | [calibration/prepare_experiment.py](calibration/prepare_experiment.py)、[calibration/train_exact.py](calibration/train_exact.py)、[calibration/model.py](calibration/model.py) | 按原题划分训练、验证和测试，拟合四分类逻辑回归，导出缩放参数、系数和偏置 |

采集时只在抽中的投机轮次根节点尝试把 `X` 放到 `B` 的位置，沿用原先两个位置的分数和节点预算。这样可以观察走过 `X` 的状态。正式实验由已拟合的分类器比较 `qX` 与 `qB`，经过原来的合法性检查，再由 Target 验证。两个过程使用相同的 70B 候选池和读出数值路径。

## 1. 准备环境和模型

解压本包，进入仓库根目录。启动方式与 [agent.md](agent.md) 相同，使用匹配的 Linux CUDA 环境。把下面的路径换成本机值。修复模式按实际模型文件做内容哈希，两个模型目录中的权重文件需要落在各自目录内。若模型来自 Hugging Face 缓存快照，检查文件是否通过符号链接指向目录外。

```bash
export TARGET_DIR=/models/Llama-3.3-70B-Instruct
export DRAFT_DIR=/models/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge
export RUN_ROOT="$HOME/oov-a100-tp8"
export GPU_LIST=0,1,2,3,4,5,6,7
mkdir -p "$RUN_ROOT"

nvidia-smi -L
python tools/install_runtime.py
python tools/install_runtime.py --verify-installed
```

已安装时只执行 `--verify-installed`。按 [agent.md](agent.md) 准备 `$RUN_ROOT/redhat8_inputs.json` 并做原生 EAGLE 的 BS2 短测。Red Hat 输入用于最终评测和训练题重合检查。

## 2. 生成 70B 对齐参数

挑一张空闲 A100。下面用物理 0 号卡举例。对齐完成前避免同时启动八卡引擎。

```bash
CUDA_VISIBLE_DEVICES=0 python -m calibration.fit_alignment \
  --target "$TARGET_DIR" --draft "$DRAFT_DIR" \
  --out "$RUN_ROOT/alignment70" --device cuda:0
```

得到 `alignment.pt`、`hot_map.json`、`alignment_report.json`。对齐矩阵是 8192×8192，hot map 记录 Draft 32K 词表映射到 Target 词表的顺序。往后的采集、拟合和修复短测使用同一份对齐参数。

## 3. 准备校准题并先跑小样本

用公开仓库附带的题目清单重建固定的校准 JSONL。先确保当前 Python 环境可导入 `pyarrow`。如果缺少该包，执行 `python -m pip install pyarrow`。从仓库根目录运行下面的命令。

```bash
python tools/rebuild_calibration_prompts.py \
  --cache-dir "$RUN_ROOT/source-cache" --output-dir "$RUN_ROOT"
```

脚本从原数据源读取指定版本，核对源文件、每道题和最终 JSONL 的 SHA-256。得到的两个 JSONL 已包含采集器直接读取的 `canonical_question_id`、`source`、`prompt`。机器离线时，可以把按清单准备的上游源文件放进缓存，或者从已有的完整实验包拷贝两个校准 JSONL 到 `$RUN_ROOT`。

| 来源 | 原数据版本 / 划分 | 短测 | 正式采集 |
|---|---|---:|---:|
| [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) | v0.1.4 | 6 | 64 |
| [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | train | 6 | 64 |
| [Hendrycks MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | 七个类别的 train | 6 | 64 |
| [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) | 上游 test，本实验用于校准 | 6 | 64 |

`calibration_pilot24.jsonl` 和 `calibration_main256.jsonl` 共 280 道不同的原题，已和冻结的 Red Hat 八个子集逐题比较。清单记录数据源的固定版本、源行 ID 和 SHA-256。正式 256 道题按原题分为 160 道训练、48 道验证、48 道内部测试。同一题的所有状态保持在同一划分。两份文件只含题目文本，分类器标签由本机 70B TP8 实际 Verify 生成。Red Hat 八个子集单独评测。

直接执行下面的命令，由 SGLang 启动八个 TP rank。输出目录每次使用新名字。

```bash
python tools/collect_tp8.py \
  --target "$TARGET_DIR" --draft "$DRAFT_DIR" \
  --prompts "$RUN_ROOT/calibration_pilot24.jsonl" \
  --eval-inputs "$RUN_ROOT/redhat8_inputs.json" \
  --alignment-dir "$RUN_ROOT/alignment70" \
  --gpus "$GPU_LIST" --bs 1 --limit 24 --max-new-tokens 96 \
  --seed 42 --x-probe-rate 0.2 \
  --output "$RUN_ROOT/collect70-smoke"
```

小样本首先检查 raw Draft/Target 输出是否出现、日志是否正常退出、两边的条件前缀能否配对。然后检查是否抓到真实的 `parent_kind=x`。如果这 24 道里面没有合适的 `X`，可以换更多独立题或提高采样率。查看候选是否被提出、是否真的进入前沿、是否最终进入验证树，区分无候选与被剪掉。不要根据最终评测表现挑校准题。

若中途退出，先看 `run.log`。原始 `draft.jsonl`、`target.jsonl` 保留了 token 前缀，采集规模扩大后会占用磁盘，和正式结果分开存放。收齐原始记录且只有最后的合并阶段需要重做时，可在新目录用 `python -m calibration.tp8_join --help` 查看离线合并参数。

此时模型仍是 TP8，BS1 指一次最多一个并发请求。采集关闭 Graph，以便观察真实张量和树行。调试记录会增加耗时，正式测速使用独立运行路径。

## 4. 扩大采集

小样本对账后，用主集的 256 道题采集，BS2，每道题最多生成 256 个 token。采集器会用目标机上的 Llama-3.3 tokenizer 重新编码，并与 `redhat8_inputs.json` 做精确 token 序列去重。

```bash
python tools/collect_tp8.py \
  --target "$TARGET_DIR" --draft "$DRAFT_DIR" \
  --prompts "$RUN_ROOT/calibration_main256.jsonl" \
  --eval-inputs "$RUN_ROOT/redhat8_inputs.json" \
  --alignment-dir "$RUN_ROOT/alignment70" \
  --gpus "$GPU_LIST" --bs 2 --limit 256 --max-new-tokens 256 \
  --seed 42 --x-probe-rate 0.2 \
  --output "$RUN_ROOT/collect70"
```

用于训练的三项主要交付物是 `exact_rows.jsonl.gz`、`runtime_contract.json`、`collection_audit.json`。原始运行证据留在同一目录。看审核中的前缀匹配、父状态和标签分布，尤其关注下面两类证据。

- 有真实的 `parent_kind=x`，表示 Draft 已经沿 `X` 往下一层计算过，而且相同条件前缀在 Target 验证树里有实际输出。
- 有 `gold == X` 和 `gold == B`，分别是当前父节点支持替换和支持保留的真实例子。前者和 `parent_kind=x` 是两个不同字段。

覆盖不足时扩大独立题数。按原题去重，多轮记录不等于更多独立问题。

## 5. 拟合并查看独立题的效果

拟合需要 NumPy 和 SciPy。先用 `python -c 'from scipy.optimize import minimize'` 验证目标环境。如果缺 SciPy，在你的实验环境内安装它，再执行拟合。

```bash
python -m calibration.prepare_experiment \
  --rows "$RUN_ROOT/collect70/exact_rows.jsonl.gz" \
  --contract "$RUN_ROOT/collect70/runtime_contract.json" \
  --alignment-dir "$RUN_ROOT/alignment70" \
  --out "$RUN_ROOT/fitted70-tp8"
```

分类器训练在 CPU 上执行。每条样本由 `p1/p2`、`B` 与 `X` 的分数关系、深度和 `has_x` 等构成 12 维输入。标签是 Target 在该前缀下的真实 Top1 属于 `A`、`B`、`X` 还是其他 token。同一道题的全部状态进入同一划分，拟合按题做样本加权。

先看 `fit/coverage.json` 和 `fit/exclusions.json`。再看 `fit/evaluation.json` 的测试集。关注 `rescue`、`lost_native2` 和 `net_local_hits`，分别代表补中、错换和净增长。若独立题的净增长持续偏负，先查数据覆盖与候选评分，再考虑正式测速。

拟合目录下 `parameters/` 包含 `alignment.pt`、`runtime_contract.json`、`calibration_asset.json`。这些文件与当前模型和 TP8 绑定，只留在目标机器。

## 6. 小批启动修复，然后做正式对照

```bash
common=(
  --target "$TARGET_DIR" --draft "$DRAFT_DIR"
  --inputs "$RUN_ROOT/redhat8_inputs.json" --gpus "$GPU_LIST"
  --methods eagle728 repair --assets "$RUN_ROOT/fitted70-tp8/parameters"
)
python tools/benchmark.py "${common[@]}" --bs 2 --limit 8 --repeats 1 \
  --output "$RUN_ROOT/repair-smoke"

python tools/benchmark.py "${common[@]}" --bs 32 --repeats 3 \
  --output "$RUN_ROOT/repair-vs-eagle70-tp8"
```

先跑小批量，检查模型身份、资产、真实 Decode Graph、Target 验证计数和来源审计。正式 BS32 时注意实际批大小和是否有跨卡候选决策错误。报告中的吞吐和耗时只读正式目录下的 `REPORT.md` 和 `results.json`。

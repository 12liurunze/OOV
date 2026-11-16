# 八卡 A100 实验运行说明

## 1. 实验对象和当前状态

| 项目 | 配置或状态 |
|---|---|
| Target | `meta-llama/Llama-3.3-70B-Instruct` |
| Draft | `lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge` |
| 设备 | 同一节点的 8 张 NVIDIA A100，单模型实例 TP8，DP1、PP1 |
| 运行负载 | Red Hat 的八个子集，每个按原顺序选 80 条，原样重复一次，每方法每轮共 1,280 个请求 |
| 原生 EAGLE-3 | 可先在目标机做基线预检、短测和三轮正式测速 |
| OOV 补词 | 已提供单独的 8192 维 TP8 移植，真实 A100 模型和跨卡 Graph 尚待验证 |
| OOV 校准 | 已提供单卡对齐、八卡 eager 采集和离线拟合工具。实际 70B TP8 校准行及分类器由目标机产出，首次启动需要完成短测 |

这里固定的是 2 号 A6000 上最后完成的补词版本源码，即保留其余三项、关闭请求槽位只读视图的版本。70B 移植在独立目录中，把读出宽度和对齐矩阵从 4096 扩展到 8192，沿用跨 TP rank 的候选归并。8B 的三项优化依赖 TP1，在 TP8 不启动。OOV 策略仍是从请求内找词表外候选，保留原生第一候选，用匹配的分类器判断是否替换第二候选，再由 Target 验证。

首次交付只需跑通原生基线。OOV 资产缺失时，补词命令会在加载模型前明确退出。正式报告中的 OOV 数字必须来自本机实测。 [官方 Draft 配置](https://huggingface.co/lmsys/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge/blob/e12dce0cccb357ca3c4d20ea5361a9b13deb0d58/config.json)

## 2. 准备环境和本地目录

在 Linux 主机上运行。使用 Python 3.11、能够运行 CUDA 13.0 PyTorch 的 NVIDIA 驱动，以及与 [requirements.runtime.txt](requirements.runtime.txt) 相容的隔离环境。该清单记录源实验的核心版本，环境需包含对应的 CUDA wheel 和系统依赖。先优先复用机器上与这些版本匹配的 SGLang 环境。源码安装器不安装或升级 Python 包。保持 8 张卡空闲，GPU 进程监控需要在能看到相同 PID 命名空间的系统环境下运行。

从 GitHub 克隆到目标机器。使用离线压缩包时，解压后进入 `OOV_A100_TP8_20261116`。在下面命令中，把模型和结果目录路径换成本机实际路径。

```bash
git clone https://github.com/12liurunze/OOV.git
cd OOV

export TARGET_DIR=/models/Llama-3.3-70B-Instruct
export DRAFT_DIR=/models/SGLang-EAGLE3-Llama-3.3-70B-Instruct-SpecForge
export RUN_ROOT="$HOME/oov-a100-tp8"
mkdir -p "$RUN_ROOT"

nvidia-smi -L
nvidia-smi topo -m
python --version
python -c 'import torch, triton, transformers; print(torch.__version__, torch.version.cuda, triton.__version__, transformers.__version__)'
python -m pip check
python tools/install_runtime.py
python tools/install_runtime.py --verify-installed
```

GitHub 版本会从 [runtime/manifest.json](runtime/manifest.json) 中指定的提交下载冻结的 SGLang 源码底包。离线压缩包已附带底包，安装默认直接读取。也可用 `--archive /local/original_runtime.tar.gz` 指定已有底包。安装会核对底包及叠加后的文件哈希。模型、Draft、对齐矩阵、分类器参数、数据集和生成输出都放在本地。

## 3. 准备八个子集

机器能访问 Hugging Face 时，在本地下载固定版本，再用 **70B Target 的 tokenizer 和聊天模板**重编码。

```bash
python tools/prepare_inputs.py --download \
  --source-dir "$RUN_ROOT/redhat-raw" --model "$TARGET_DIR" \
  --output "$RUN_ROOT/redhat8_inputs.json"
```

已有原始文件时去掉 `--download`，保留同一 `--source-dir` 参数。工具会校验原始文件哈希。有明确的归档 8B tokenizer 指纹差异时，可加 `--allow-tokenization-drift` 重新执行。生成的 manifest 会记录 70B 实际 token 序列哈希及与归档的差异。各方法必须共用这个文件。

```bash
python tools/prepare_inputs.py --check-prepared "$RUN_ROOT/redhat8_inputs.json"
```

如果准备阶段允许了 tokenizer 差异，检查时也加同一开关。`question` 和 `writing` 使用同一组题，报告另外给出去重汇总。

## 4. 预检并跑原生 EAGLE

脚本从一个 Python 进程启动八个 rank。直接运行下面的命令。机器上的物理 GPU 编号以 `nvidia-smi` 为准。

```bash
common=(
  --target "$TARGET_DIR" --draft "$DRAFT_DIR"
  --inputs "$RUN_ROOT/redhat8_inputs.json"
  --gpus 0,1,2,3,4,5,6,7
)

python tools/benchmark.py "${common[@]}" --preflight \
  --output "$RUN_ROOT/eagle70-tp8"

python tools/benchmark.py "${common[@]}" --methods eagle728 \
  --bs 2 --limit 8 --repeats 1 --output "$RUN_ROOT/eagle-smoke"

python tools/benchmark.py "${common[@]}" --methods eagle728 \
  --bs 32 --repeats 3 --output "$RUN_ROOT/eagle70-tp8"
```

预检核对模型配置、文件和 8 张 A100 的枚举，模型加载及 Graph 验收在短测发生。脚本检查 Decode 达到真实目标 BS 且满批启用 Graph。固定 BF16、EAGLE3、S7 / Top2 / N8、greedy、种子 42、512 输出 token 上限、2048 上下文。Radix Cache 和 Decode Graph 开启，Prefill Graph 关闭。每个子集测速前用同一组唯一输入初始化 Radix Cache。正式表统计整个子集的端到端耗时，实际 BS 分布另记在每轮数据里。

短测输出用于验证环境、权重、TP 和 Graph。正式数据请使用 BS32 的目录。失败日志保存在各方法目录下，具备相同参数时可以原样重跑。脚本复用验证通过的轮次并保留中断目录。显存不足时，两种方法使用同一 `--mem-fraction` 并换一个新的结果目录。出现外部计算进程时，脚本退出并仅清理自身子进程。运行时长请以本机短测和第一轮为准。

## 5. 补词版的资产和验收

按 [CALIBRATION_TP8.md](CALIBRATION_TP8.md) 在目标机器上生成 8192×8192 的词表对齐矩阵，用 `tools/collect_tp8.py` 收集同一组 Target、Draft、TP8 数值路径和候选策略下的真实 Draft 特征及 Target 标签，再拟合分类器。先在目标机通过采集短测，评测题保持独立。

已有真实 TP8 校准行、合同并完成资产拟合后，`OOV_ASSETS` 指向含有 `alignment.pt`、`runtime_contract.json` 和 `calibration_asset.json` 的 `parameters` 目录。合同中的模型路径应解析到当前本机实际加载的位置。权重文件需要在各自模型目录内。Hugging Face 缓存目录如果通过符号链接指向目录外的 blobs，请使用已物化的 checkpoint 目录。

```bash
export OOV_ASSETS="$RUN_ROOT/fitted70-tp8/parameters"

python tools/stage_70b.py
python tools/benchmark.py "${common[@]}" --methods eagle728 repair \
  --assets "$OOV_ASSETS" --bs 2 --limit 8 --repeats 1 \
  --output "$RUN_ROOT/repair-smoke"

python tools/benchmark.py "${common[@]}" --methods eagle728 repair \
  --assets "$OOV_ASSETS" --bs 32 --repeats 3 \
  --output "$RUN_ROOT/repair-vs-eagle70-tp8"
```

运行时核对本机的模型权重、tokenizer、32K 映射、对齐矩阵、校准合同和实际加载的模块。完整权重哈希在 TP rank 初始化期间需要时间。短测应通过跨 rank 候选一致性、真实 Graph batch 和补词来源计数检查，之后再进入正式对照。

## 6. 拿到哪些结果

| 文件 | 内容 |
|---|---|
| `plan.json` | GPU、模型配置与权重分片清单、输入哈希、校准资产哈希、运行时和软件版本 |
| `rN/<method>/metrics.json` | 八个子集每个方法这一轮的请求数、端到端耗时、吞吐、接受率、接受长度、输出 token、上限命中、实际 BS、Graph 步数和来源计数 |
| `rN/<method>/<subset>.outputs.json` | 逐请求输出哈希、停止原因和接受统计 |
| `rN/<method>/measure_<subset>.jsonl` | 调度步、真实批大小、Graph 状态、该步提交 token 和接受数 |
| `results.json` | 各子集三轮明细、均值、样本标准差、整体和去重汇总。有两个方法时也记录补词相对原生的变化 |
| `REPORT.md` | 方便发送的八子集结果表 |
| `rN/<method>/run.log` | 环境、预热、生成或失败日志 |

吞吐等于生成 token 总数除以子集端到端秒数。接受率等于接受的草稿数除以提出的草稿数，接受长度等于接受的草稿数除以验证轮数，两者都排除 Target 补发 token。报告同时列出输出 token 数和耗时，便于判断吞吐变动是否伴随输出变长。输出哈希用于复核生成轨迹，数学或代码答对率需单独评估。

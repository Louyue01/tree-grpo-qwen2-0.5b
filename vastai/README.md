# 在 Vast.ai 上启动 Tree-GRPO 训练（Multi-hop QA · Qwen2.5-0.5B）

本目录是「镜像 + 脚本」打包方案：镜像里装好全部依赖（两个 conda 环境），代码与启动脚本已随镜像
内置到 `/workspace/Tree-GRPO`。你只需租卡、构建/推送镜像、按顺序跑几个脚本。

## 0. 你需要准备

- 一个 **Docker Hub** 账号（或其它镜像仓库），用于推送镜像。
- 一个 **GitHub** 账号 + 一个你自己的仓库（放代码，用来做本地 ↔ 实例的同步；见 §8）。
- Vast.ai 账号，并已充值。

## 1. 构建并推送镜像

在**本仓库根目录**（Windows 或任意带 Docker 的机器）执行：

```bash
docker build -f docker/Dockerfile -t <你的dockerhub用户名>/tree-grpo:qwen2.5-0.5b .
docker push <你的dockerhub用户名>/tree-grpo:qwen2.5-0.5b
```

> 构建要点：
> - 基础镜像 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`，flash-attn 会从源码编译（约 10–20 分钟，属正常）。
> - 两个 conda 环境：`treegrpo`（torch 2.6.0 + vllm 0.8.5.post1 + flash-attn + verl）和
>   `retriever`（faiss-gpu + e5 + uvicorn/fastapi）。
> - 镜像内含本仓库代码；也可以在租卡后挂载你自己的代码目录覆盖 `/workspace/Tree-GRPO`。

## 2. 租卡（选机）

- **GPU**：1× RTX 4090（24GB）即可；A10G / L4 / L40S 更宽裕。0.5B 全参 + vLLM rollout + FSDP offload + 检索服务器，24GB 足够。
- **磁盘**：**≥ 100GB**（wiki-18 索引 + 语料约 10–20GB，Qwen2.5-0.5B + 教师模型 7B-Instruct + e5 约 15GB，checkpoint 与数据集再留余量）。租用时把磁盘调大，或额外挂一块 volume。
- **镜像**：填你第 1 步推送的镜像地址。
- **按需 vs 可中断**：训练建议选 **On-demand**（可中断实例便宜但可能被回收；可中断适合冒烟测试）。
- 需要**外网**以下载模型/数据（Vast.ai 实例默认有外网）。

## 3. 首次登入

SSH 进实例后。镜像只负责**装环境**，代码统一用 git 管：clone 你自己的仓库到
`/workspace/Tree-GRPO`（**必须用这个原路径**，否则镜像里 `pip install -e .` 的 editable 安装会失效）：

```bash
source /opt/miniforge3/etc/profile.d/conda.sh
cd /workspace
rm -rf Tree-GRPO                              # 删掉镜像里烘焙的旧代码
git clone https://github.com/<你的用户名>/tree-grpo-qwen0.5b.git Tree-GRPO
cd Tree-GRPO
conda activate treegrpo
pip install -e .                              # editable 路径未变，跑一下更保险
```

> - 仓库是 **public** 则直接 clone；**private** 见 §8.3 的 PAT 认证。
> - `rm -rf` 只删代码；如果已经跑过 `download_data.sh`，先把 `models/ data/ retrieval/` 挪走再删。
> - 建议在第一次下载数据**之前**就做这一步，最干净。

## 4. 跑通完整流程

### 4.1 下载模型 + 检索索引 + 处理数据集（`treegrpo` 环境）

```bash
conda activate treegrpo
bash setup/download_data.sh
```

产出：
- `models/Qwen2.5-0.5B`、`models/Qwen2.5-7B-Instruct`（教师）、`models/e5-base-v2`
- `retrieval/e5_Flat.index`、`retrieval/wiki-18.jsonl`
- `data/train.parquet`、`data/test.parquet`

> 所有模型（Qwen2.5-0.5B / 7B-Instruct、e5）与 FlashRAG 数据集都是公开的，无需 HF token。
> 若替换成 gated 模型，先 `huggingface-cli login`。

### 4.2 启动检索服务器（`retriever` 环境，另开一个终端/tmux）

```bash
conda activate retriever
bash setup/launch_retriever.sh
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即就绪。

### 4.3 生成 SFT 格式预热数据（`treegrpo` 环境，第三个终端）

```bash
conda activate treegrpo
python scripts/make_sft_data.py \
    --teacher models/Qwen2.5-7B-Instruct \
    --n_samples 1000 \
    --out data/sft.parquet \
    --retriever_url http://127.0.0.1:8000/retrieve
```

教师模型对 HotpotQA 训练集抽样，跑 ReAct 循环，产出 turn 级 `(prompt, response)` 对。
（0.5B base 从零学 `<think>/<search>/<answer>` 标签很难，这一步先教格式。）

### 4.4 SFT 格式预热

```bash
conda activate treegrpo
bash run_sft.sh
```

输出 HF 格式 checkpoint 到 `sft_checkpoint/`。

### 4.5 RL 训练（GRPO 基线 or Tree-GRPO）

```bash
conda activate treegrpo

# GRPO 基线
BASE_MODEL=$PWD/sft_checkpoint/global_step_<N> bash run_grpo.sh

# Tree-GRPO（树搜索）
BASE_MODEL=$PWD/sft_checkpoint/global_step_<N> bash run_tree.sh
```

> `<N>` 是 SFT 最后一个 checkpoint 的步数；`ls sft_checkpoint/` 查看。

## 5. 冒烟测试（强烈建议先做）

正式训练前先用极小规模确认整条链路（retriever → rollout → advantage → loss → checkpoint）：

```bash
# 数据只取几十条 + 步数极小
TOTAL_TRAINING_STEPS=2 bash run_tree.sh
```

能正常打印 rollout / loss 并保存 checkpoint 后，再恢复 `TOTAL_TRAINING_STEPS=180`（默认）正式跑。

## 6. 常用环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 单卡默认用 0 号卡 |
| `BASE_MODEL` | `models/Qwen2.5-0.5B` | 训练起点（SFT 后改成 `sft_checkpoint/global_step_<N>`） |
| `DATA_DIR` | `data` | `train.parquet`/`test.parquet` 所在目录 |
| `RETRIEVER_URL` | `http://127.0.0.1:8000/retrieve` | 检索服务地址 |
| `EXPERIMENT_NAME` | 见脚本 | checkpoint / 日志名 |
| `LR` | `1e-5` | RL 学习率（原脚本此处是未定义变量，已修复） |
| `TOTAL_TRAINING_STEPS` | `180` | 训练步数（冒烟测试改小） |

## 7. 常见坑

- **flash-attn / vLLM 显存**：脚本里 `actor_rollout_ref.rollout.gpu_memory_utilization=0.6`。
  若 OOM，把它降到 `0.45`，或把 `ppo_micro_batch_size`/`log_prob_micro_batch_size` 从 `2` 降到 `1`。
- **注意力后端**：原脚本 `VLLM_ATTENTION_BACKEND=XFORMERS`，但 README 安装流程并不装 xformers，
  已改为默认 `FLASH_ATTN`（镜像里已装 flash-attn）。如需 xformers，需自行 `pip install xformers` 并
  用 `VLLM_ATTENTION_BACKEND=XFORMERS` 覆盖。
- **swanlab**：原脚本用 `trainer.logger=['swanlab','console']` 需 API key；已改为 `['console']`。
- **`$lr` 未定义**：原 `train_multihopqa_tree_search.sh` 引用未定义的 `$lr`，已修复并暴露为 `LR` 环境变量。
- **ray job submit**：已改为直接 `python3 -m verl.trainer.main_ppo_format_ts`（模块内部自行 `ray.init()`），
  无需先起 ray head。
- **faiss-gpu**：PyPI 上 `faiss-gpu` 到 1.7.2 为止；镜像改装 `faiss-gpu-cu12==1.7.3`（`import faiss` 不变）。
  若在别处安装失败，可退回 `conda install -c conda-forge faiss-gpu=1.7.3`。
- **检索服务器没起来**：先确认 `retrieval/e5_Flat.index`、`retrieval/wiki-18.jsonl`、`models/e5-base-v2`
  三者存在（`setup/download_data.sh` 的产出）。`curl -s -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}'` 应返回 JSON。

## 8. 迭代同步（git 方式）

本地改代码后不用重新构建镜像，走 git 循环即可。只有改 `docker/Dockerfile`、`pyproject.toml`
（依赖）或 conda 环境时才需要重新 build + push + 重新租。

### 8.1 本地首次提交并推送（一次性）

```powershell
cd "F:\Study Files\Projects\GRPO-learning\Tree-GRPO"
git add -A
git status                      # 确认没有 models/ data/ retrieval/ 等大文件
git commit -m "Reproduce Tree-GRPO (Qwen2.5-0.5B + SFT warmup)"
git remote add mine https://github.com/<你的用户名>/tree-grpo-qwen0.5b.git
git push -u mine main
```

> `origin` 仍指向上游 `AMAP-ML/Tree-GRPO`（将来拉上游更新用），你自己的仓库叫 `mine`。
> 仓库里已带 `.gitignore`，排除了 `models/ data/ retrieval/ *.parquet *.index *.jsonl` 等大文件。

### 8.2 迭代循环

本地改代码 → commit → push：

```powershell
cd "F:\Study Files\Projects\GRPO-learning\Tree-GRPO"
git add -A
git commit -m "fix: 调整单卡 batch / 修复 xxx"
git push mine main
```

实例上拉取并重跑：

```bash
cd /workspace/Tree-GRPO
git pull origin main     # clone 出来的仓库 origin 默认就是你的仓库
bash run_tree.sh         # 或其它命令重新跑
```

改 `.py` 下次运行自动生效（editable 安装）；改 `.sh` 直接生效；改检索服务器代码则重启该进程。

### 8.3 私有仓库的 clone 认证（public 跳过）

1. GitHub → Settings → Developer settings → **Personal access tokens → Tokens (classic)** →
   Generate new token，勾选 **`repo`** 权限。
2. clone 时把 token 当密码用，或写进 URL：
   ```bash
   git clone https://<用户名>:<PAT>@github.com/<用户名>/tree-grpo-qwen0.5b.git Tree-GRPO
   ```
3. 让后续 `git pull` 记住凭据：
   ```bash
   git config --global credential.helper store
   git pull      # 第一次输入用户名+PAT 后不再提示
   ```

## 9. 预期结果（复现参照）

论文（Qwen2.5-3B）Multi-hop 平均 EM：GRPO 31.8 → Tree-GRPO 36.8。
0.5B 无官方基线：复现目标是看 **Tree-GRPO 相对 GRPO 的提升趋势**是否出现（而非绝对 EM 对齐）。
算法级改进（引导式树扩展 / 步级优势等）在复现跑通后另行规划。

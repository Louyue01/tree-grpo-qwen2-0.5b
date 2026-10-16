# 在 Vast.ai 上启动 Tree-GRPO 训练（Multi-hop QA · Qwen2.5-0.5B-Instruct）

本目录是「官方 CUDA 镜像 + 一条安装脚本」方案：租一个带 CUDA 12.4 **devel** 工具链的裸机镜像，
SSH 进去 `git clone` 代码后跑 `bash setup/install.sh` 装好全部依赖（两个 conda 环境），再按顺序跑
下载数据 → 起检索 → SFT 预热 → RL 训练。**不构建、不推送 Docker 镜像**。

## 0. 你需要准备

- 一个 **GitHub** 账号 + 一个你自己的仓库（放代码，用来做本地 ↔ 实例的同步；见 §7）。
- Vast.ai 账号，并已充值。

## 1. 租卡（选机）

- **镜像/模板**：必须含 **CUDA 12.4 devel 工具链（有 nvcc）**，因为 `install.sh` 会从源码编译
  flash-attn。推荐直接用官方 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`（Vast.ai 选机时在
  "Edit image & config" 里把镜像填成这个地址即可；或选一个带 nvcc 的 CUDA 12.x 模板）。
- **GPU**：1× RTX 4090（24GB）即可；A10G / L4 / L40S 更宽裕。0.5B 全参 + vLLM rollout + FSDP offload
  24GB 足够（检索走 CPU，不占显存）。
- **内存**：**≥ 96GB（建议 128GB）**。wiki-18 的 FAISS 索引约 60GB 会被检索服务器整块读进内存
  （`faiss.read_index`，CPU 检索），再加训练进程约 20GB，内存不足会 OOM。选机时务必看 RAM 参数。
- **磁盘**：**≥ 200GB**。wiki-18 索引 60GB（part_aa 40G + part_ab 20G，合并时瞬时再 +60G）+
  语料 gz 5G / 解压后约 18G + 模型 18G（含 7B 教师）+ 两个 conda 环境约 17G + checkpoint，
  峰值约 160–180G。**100G 装不下**。
- **按需 vs 可中断**：训练建议选 **On-demand**（可中断实例便宜但可能被回收；可中断适合冒烟测试）。
- 需要**外网**以下载模型/数据（Vast.ai 实例默认有外网）。

## 2. 首次登入 + 装依赖

SSH 进实例后：

```bash
cd /workspace
git clone https://github.com/<你的用户名>/tree-grpo-qwen0.5b.git Tree-GRPO
cd Tree-GRPO
bash setup/install.sh        # 装 miniforge + treegrpo/retriever 两个 conda 环境，约 20–40 分钟
```

> - 仓库是 **public** 则直接 clone；**private** 见 §7.3 的 PAT 认证。
> - `install.sh` 会把 conda 装到 `/opt/miniforge3`（root 用户时），并 `conda init bash`。
>   装完后**开一个新 shell**（或 `source /opt/miniforge3/etc/profile.d/conda.sh`）让 `conda activate` 生效。
> - 脚本有完成标记（`~/.cache/treegrpo-install/`），中途失败重跑会跳过已装好的环境；
>   想彻底重装用 `FORCE=1 bash setup/install.sh`。
> - 依赖版本都写死在脚本里，和镜像方案一致：treegrpo = py3.12.9 / torch 2.6.0 / vllm 0.8.5.post1 /
>   transformers 4.53.2 / tensordict 0.7.0 / flash-attn；retriever = py3.10.13 / torch 2.6.0 / faiss-gpu-cu12 1.10.0 / uvicorn / fastapi。

## 3. 跑通完整流程

### 3.1 下载模型 + 检索索引 + 处理数据集（`treegrpo` 环境）

```bash
conda activate treegrpo
bash setup/download_data.sh
```

产出：
- `models/Qwen2.5-0.5B-Instruct`、`models/Qwen2.5-7B-Instruct`（教师）、`models/e5-base-v2`
- `retrieval/e5_Flat.index`、`retrieval/wiki-18.jsonl`
- `data/train.parquet`、`data/test.parquet`

> 所有模型（Qwen2.5-0.5B / 7B-Instruct、e5）与 FlashRAG 数据集都是公开的，无需 HF token。
> 若替换成 gated 模型，先 `huggingface-cli login`。

### 3.2 启动检索服务器（`retriever` 环境，另开一个终端/tmux）

```bash
conda activate retriever
bash setup/launch_retriever.sh
```

看到 `Uvicorn running on http://0.0.0.0:8000` 即就绪。

### 3.3 生成 SFT 格式预热数据（`treegrpo` 环境，第三个终端）

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

### 3.4 SFT 格式预热

```bash
conda activate treegrpo
bash run_sft.sh
```

输出 HF 格式 checkpoint 到 `sft_checkpoint/`。

### 3.5 RL 训练（GRPO 基线 or Tree-GRPO）

```bash
conda activate treegrpo

# GRPO 基线
BASE_MODEL=$PWD/sft_checkpoint/global_step_<N> bash run_grpo.sh

# Tree-GRPO（树搜索）
BASE_MODEL=$PWD/sft_checkpoint/global_step_<N> bash run_tree.sh
```

> `<N>` 是 SFT 最后一个 checkpoint 的步数；`ls sft_checkpoint/` 查看。

## 4. 冒烟测试（强烈建议先做）

正式训练前先用极小规模确认整条链路（retriever → rollout → advantage → loss → checkpoint）：

```bash
# 数据只取几十条 + 步数极小
TOTAL_TRAINING_STEPS=2 bash run_tree.sh
```

能正常打印 rollout / loss 并保存 checkpoint 后，再恢复 `TOTAL_TRAINING_STEPS=180`（默认）正式跑。

## 5. 常用环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `CUDA_VISIBLE_DEVICES` | `0` | 单卡默认用 0 号卡 |
| `BASE_MODEL` | `models/Qwen2.5-0.5B-Instruct` | 训练起点（SFT 后改成 `sft_checkpoint/global_step_<N>`） |
| `DATA_DIR` | `data` | `train.parquet`/`test.parquet` 所在目录 |
| `RETRIEVER_URL` | `http://127.0.0.1:8000/retrieve` | 检索服务地址 |
| `EXPERIMENT_NAME` | 见脚本 | checkpoint / 日志名 |
| `LR` | `1e-5` | RL 学习率（原脚本此处是未定义变量，已修复） |
| `TOTAL_TRAINING_STEPS` | `180` | 训练步数（冒烟测试改小） |

## 6. 常见坑

- **nvcc 缺失导致 flash-attn 编译失败**：这是最常见的一坑。租卡时务必选 **devel** 镜像
  （含 CUDA toolkit），别选纯 runtime 镜像。`install.sh` 开头会检测 `nvcc` 并给出提示。
- **flash-attn / vLLM 显存**：脚本里 `actor_rollout_ref.rollout.gpu_memory_utilization=0.6`。
  若 OOM，把它降到 `0.45`，或把 `ppo_micro_batch_size`/`log_prob_micro_batch_size` 从 `2` 降到 `1`。
- **注意力后端**：原脚本 `VLLM_ATTENTION_BACKEND=XFORMERS`，但安装流程并不装 xformers，
  已改为默认 `FLASH_ATTN`（flash-attn 是实际安装的依赖）。如需 xformers，自行 `pip install xformers` 并
  用 `VLLM_ATTENTION_BACKEND=XFORMERS` 覆盖。
- **Qwen2.5 base 没有 chat_template（会导致奖励格式判 0）**：多跳 QA 的奖励函数
  `qa_em_format.is_valid_sequence` 要求解码出的 `prompt+response` 里含 `<|im_start|>assistant` 标记；
  而 `rl_dataset.py` 只有在 `tokenizer.chat_template` 存在时才会把 prompt 包成 ChatML。**本项目默认用
  Instruct（自带 ChatML 模板），此坑自动规避**；仅在切回 base 模型时才需要 `download_data.sh` 里的
  `scripts/add_chat_template.py` 把教师的 ChatML 模板拷到 base（否则格式奖励恒为 0）。
- **swanlab**：原脚本用 `trainer.logger=['swanlab','console']` 需 API key；已改为 `['console']`。
- **`$lr` 未定义**：原 `train_multihopqa_tree_search.sh` 引用未定义的 `$lr`，已修复并暴露为 `LR` 环境变量。
- **ray job submit**：已改为直接 `python3 -m verl.trainer.main_ppo_format_ts`（模块内部自行 `ray.init()`），
  无需先起 ray head。
- **faiss-gpu**：PyPI 上 `faiss-gpu-cu12` 的 1.7.x 已下架，现在从 1.8.x 起；改装 `faiss-gpu-cu12==1.10.0`（`import faiss` 不变）。
  若在别处安装失败，可退回 `conda install -c conda-forge faiss-gpu`（不指定版本，取最新）。
- **检索服务器没起来**：先确认 `retrieval/e5_Flat.index`、`retrieval/wiki-18.jsonl`、`models/e5-base-v2`
  三者存在（`setup/download_data.sh` 的产出）。`curl -s -X POST http://127.0.0.1:8000/retrieve -H 'Content-Type: application/json' -d '{"queries":["test"],"topk":1}'` 应返回 JSON。

## 7. 迭代同步（git 方式）

本地改代码后走 git 循环即可，**不用重装依赖**。只有改 `setup/install.sh`、`pyproject.toml`（依赖）时
才需要在新实例上重跑安装（或在原实例上手动补装对应包）。

### 7.1 本地首次提交并推送（一次性）

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

### 7.2 迭代循环

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

### 7.3 私有仓库的 clone 认证（public 跳过）

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

## 8. 预期结果（复现参照）

论文（Qwen2.5-3B）Multi-hop 平均 EM：GRPO 31.8 → Tree-GRPO 36.8。
0.5B 无官方基线：复现目标是看 **Tree-GRPO 相对 GRPO 的提升趋势**是否出现（而非绝对 EM 对齐）。
算法级改进（引导式树扩展 / 步级优势等）在复现跑通后另行规划。

# Tree-GRPO 复现进展摘要（Qwen2.5-0.5B 多跳 QA）

> 本文件是给新 Claude 会话 / 人工交接用的状态快照。远程实例：Vast.ai `root@C.56064409:/workspace/Tree-GRPO`。

## 目标
复现 Tree-GRPO（ICLR 2026, arXiv:2509.21240），底座 Qwen2.5-0.5B-Instruct，HotpotQA 训练 + 2Wiki/MuSiQue/Bamboogle 评测。流程：SFT warmup → GRPO/Tree-GRPO RL。

## 已完成
1. SFT warmup，checkpoint 在 `/workspace/Tree-GRPO/sft_checkpoint/global_step_825`。
2. 一轮 180-step Tree-GRPO RL，训练脚本 `train_multihopqa_tree_search.sh`。
3. 修复了 tree-search rollout 的 vLLM 崩溃：`verl/workers/rollout/vllm_rollout/vllm_rollout_spmd_ts.py` 的 `_pre_process_inputs` 做了鲁棒化（防 `TypeError: prompt must be ...`），commit `bc695ec`。训练因此全程无崩溃。

## RL 结果：发散
验证指标是**纯 EM（精确匹配）准确率**（不是带格式分的 reward——验证用 `qa_em_format.compute_score_em` 且 `val_reward_fn` 没传格式分，格式分全为 0）。

| 指标 | step 60 | step 120 | step 180 |
|---|---|---|---|
| actor/kl_loss | 0.566 | 4.427 | 更高 |
| 2Wiki EM | 22.2% | 9.3% | 0.0% |
| HotpotQA EM | 15.8% | 2.6% | 0.0% |
| MuSiQue EM | 2.7% | 0.0% | 0.0% |
| Bamboogle EM | 0.0% | 0.0% | 0.0% |

KL 从 0.566 爆到 4.427，EM 单调崩到 0，模型输出退化成 token 汤。根因：`kl_loss_coef=0.001` 太弱 + 0.5B 底座太弱。

## SFT 基线（val_only 评估）
`EXPERIMENT_NAME=sft-baseline-eval`，日志 `verl_log/sft-baseline-eval.log`。
- HotpotQA：**8.77%**
- MuSiQue：**2.70%**
- 2Wiki：**14.52%**
- Bamboogle：**0.0%**

## 最终结论
- RL **不是纯负收益**：HotpotQA 从 SFT 8.77% → step 60 15.8%（+7.0pp），2Wiki 从 14.52% → 22.2%（+7.7pp），MuSiQue 持平（2.7%），Bamboogle 两边均为 0。
- **最终模型 = step 60 checkpoint**（≥ SFT 全部四个集，发散路径上但尚未崩）。
- 问题是**后段发散**（KL 爆炸），step 120/180 的 checkpoint 已废。

## 待办
1. ~~补 SFT 基线的 2wiki / bamboogle 两个 EM 值~~（已完成：2Wiki 14.52%，Bamboogle 0.0%）。
2. ~~定最终模型~~（已完成：step 60 ≥ SFT，采用 step 60）。
3. 重训（方案 A 已定，参数已改入脚本）：从 SFT 重跑（`sft_checkpoint/global_step_825`），不续 step 60——`kl_loss_coef=0.02`、`lr=3e-6`、`total_training_steps=160`、`save_freq`/`test_freq=40`、`val_before_train=true`、`val_data_num=100`。跑时覆盖 `BASE_MODEL`；盯 `actor/kl_loss` 是否在 40 步后稳住。

## 环境注意
- 训练/评估都依赖 retriever（`http://127.0.0.1:8000/retrieve`，独立进程，`retrieval_server.py`），跑 val_only 前要确认它还活着：`ps aux | grep retrieval_server | grep -v grep`。
- 当前后台会话的终端有问题（滚轮切历史、多行粘贴只粘最后一行），是后台 job 的 TTY 退化，无法会话内修复；交互用全新 `claude` 前台会话。

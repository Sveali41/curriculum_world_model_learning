# Stochastic MiniGrid 修改记录

## 目的

在现有 `CustomMiniGridEnv` 的固定地图、观测编码、奖励和动作接口不变的前提下，引入第一版 stochastic dynamics，用于后续 stochastic world-model 和策略鲁棒性实验。

当前版本实现导航动作随机失效，并为 MiniGrid WM 增加通用的离散
stochastic latent。统一开关会同时控制真实环境、数据、WM 训练和共享的
PPO/MPC/MCTS imagined rollout。

## Stochastic 定义

导航动作 `left`、`right`、`forward` 的执行规则为：

\[
P(a_{\mathrm{executed}}=a_{\mathrm{requested}})=1-p
\]

\[
P(a_{\mathrm{executed}}=\mathrm{NOOP})=p
\]

失败时使用环境内部的 `NOOP` outcome，不调用也不暴露 MiniGrid 原生 `done` action。该 outcome 不会改变 agent pose 或物体状态，但会消耗一个 timestep。

`pickup`、`toggle` 和 `drop` 在当前版本中始终正常执行。

## 环境 API 修改

文件：`wm/domain/minigrid/minigrid_custom_env.py`

新增构造参数：

```python
stochastic_enabled: bool = False
move_failure_prob: float = 0.2
```

`move_failure_prob` 必须属于 `[0, 1]`。每次 `step()` 都会在启用 stochastic 且动作属于导航动作时，用环境自己的 `self.np_random` 采样一次。

每一步的 `info` 都包含：

```python
{
    "requested_action": int,      # 请求执行的 native action
    "executed_action": int,       # 实际执行的 native action；失败时为 -1 (internal NOOP)
    "action_failed": bool,
    "stochastic_enabled": bool,
    "move_failure_prob": float,
}
```

环境、策略和数据集对外的 action space 都严格保留原来的 6 个动作。内部 `NOOP=-1` 只是 `info["executed_action"]` 的诊断标记，不是 MiniGrid/native action，也不加入 action space 或 dataset action；传入 action `6` 会被拒绝。

## Reset 与随机数

MiniGrid 环境中的随机起点、随机方向和动作失效统一使用 `self.np_random`，因此 `reset(seed=...)` 可以复现初始状态和后续 stochastic sequence。

起点规则为：

| 条件 | 起点行为 |
|---|---|
| 显式 `agent_start_pos` | 使用显式位置 |
| 有 `S` 标记且未替换 | 使用第一个 `S` |
| `replace_start_with_empty=True` | 从空格中随机选择，供 uniform collection 使用 |
| 无上述条件 | 抛出明确的起点配置错误 |

未指定 `agent_start_dir` 时，普通模式和 uniform 模式都会随机选择初始方向。

## 配置

以下配置加入了 `domains.minigrid.stochastic`：

- `trainer/conf/config_mac.yaml`
- `trainer/conf/config_dr.yaml`
- `trainer/conf/config_cl.yaml`
- `trainer/conf/config_target_baseline.yaml`
- `wm/modelBased/config/config.yaml`

默认配置为：

```yaml
stochastic:
  enabled: false
  move_failure_prob: 0.2
```

开启实验时设置：

```yaml
domains:
  minigrid:
    stochastic:
      enabled: true
      move_failure_prob: 0.2
```

`domains.minigrid.stochastic.enabled` 是唯一需要修改的用户开关。开启后，
环境、数据采集、真实 PPO 训练/评估和 MiniGrid WM 会共同使用 stochastic
配置；WM 自动选择 `latent_v2`。关闭后 WM 自动选择 deterministic `none`
路径。代码仍接受 `attention_model.stochastic_model` 和
`stochastic_latent.enabled`，仅用于旧 checkpoint/研究 ablation 的兼容；普通
实验不需要重复设置。

缺少该配置块的旧配置会回退到 `enabled=false`。

## 入口接入

`stochastic_env_kwargs(cfg)` 位于 `wm/domain/minigrid/minigrid_support.py`，负责读取统一配置并生成环境构造参数。

当前已接入：

- MiniGrid support 的 text/layout 环境构造；
- 数据采集环境；
- trainer 生成 mini-task 的环境；
- PPO 真实环境训练；
- PPO 真实环境评估；
- pipeline 子进程的 domain/policy 配置传递。

A*/Dijkstra 和 planner warm-start 仍保持原有确定性构造；使用共享 WM rollout
的 PPO/MPC/MCTS 会在开关开启时从 prior latent 采样。

## Dataset identity

`wm/modelBased/common/artifacts.py` 的 MiniGrid identity 现在记录：

```json
{
  "stochastic": {
    "enabled": false,
    "move_failure_prob": 0.2
  }
}
```

这样不同 stochastic transition kernel 的数据集不会被无提示地混用。

## 当前 WM outcome baseline

文件：`wm/modelBased/world_model/AttentionWM.py`、
`wm/modelBased/world_model/AttentionWM_support.py`

当前实现是在现有 success transition decoder 旁增加一个二分类 outcome head：

```text
execute      -> 使用现有 state/inventory transition heads
noop_failure -> 完整复制当前 state/inventory
```

模型输出的接口为：

```python
prediction = model.forward_stochastic(obs, action, info, inv=inventory)
# prediction["outcome_probs"]: [batch, 2]
# prediction["state_logits"], prediction["inventory_logits"]: success branch
```

`action_failed` 作为 outcome supervision；失败样本不进入普通 transition
loss，避免把成功和失败混成一个 KEEP 分布。outcome loss、failure rate 和
outcome accuracy 会写入训练/验证指标。

该版本是 MiniGrid 专用的 supervised outcome baseline。它可以预测
`execute/noop_failure` 的概率，但 outcome 没有作为 embedding 条件化 decoder，
也没有通用 latent prior/posterior，因此不等同于完整的 stochastic latent WM。

配置默认关闭：

```yaml
attention_model:
  stochastic_latent:
    enabled: false
    loss_weight: 1.0
```

启用时必须使用带 `action_failed` 的自然 transition 数据，并关闭
`transition_balanced_sampling`，否则 outcome 概率会被重采样分布污染。
同时应使用单独的 stochastic latent checkpoint 路径；旧 deterministic
checkpoint 会因 contract 不匹配而拒绝加载。

## 通用 stochastic latent v2 实现

当前已在 MiniGrid 的 AttentionWM encoder 和 decoder 之间加入共享的 one-step
discrete latent layer：

```text
state + action -> prior p(z|s,a) -> sample z
state + action + next_state -> posterior q(z|s,a,s')   # 仅训练
encoded state + latent embedding -> domain decoder -> next state distribution
```

`z` 使用可配置的 multi-categorical 表示 `[num_factors, num_classes]`。MiniGrid
第一版使用一个二分类 factor，并用 `action_failed` 对它提供可选语义监督；以后
Crafter 可以增加多个 factors 表示多个实体或事件，而不改变公共接口。

粗略接口为：

```python
prediction = model.forward_distribution(
    state,
    action,
    inventory=None,
    next_state=None,      # 提供时同时计算 posterior，训练使用
    next_inventory=None,
    latent_labels=None,   # 可选监督，例如 action_failed
    sample_mode="sample", # sample | mode
)

# prediction:
# prior_logits, posterior_logits, latent_sample
# state_logits, inventory_logits
```

训练使用 latent-conditioned transition loss，加上 posterior/prior KL；存在明确
事件标签时再增加 auxiliary categorical loss。已标注的 factor 可以直接使用标签
作为训练 sample，未标注 factor 才从 posterior 采样。预测和 imagined rollout
只能从 prior 采样，不能读取 `next_state` 或 `action_failed`。

当前实现状态：

1. `outcome_v1` 保留为独立 baseline，`latent_v2` 使用独立 model/checkpoint contract。
2. v2 已加入 prior、posterior、latent embedding 和 latent-conditioned decoder。
3. v2 支持自然 transition sampling、KL loss、可选 `action_failed` 辅助监督。
4. `forward_distribution()` 支持 prior-only sampling、posterior training 和 mode prediction；
   共享 `minigrid_wm_rollout` 在 stochastic 开启时使用 prior sampling，供
   PPO/MPC/MCTS imagined rollout 使用。
5. 已加入接口、shape、seed sampling、latent label 和反向传播 smoke tests。

v2 的内部超参数如下（普通实验无需重复开启模型开关）：

```yaml
attention_model:
  stochastic_latent:
    num_factors: 1
    num_classes: 2
    kl_weight: 0.1
    supervision_weight: 1.0
```

v2 当前只接入 MiniGrid。Crafter 接入前仍需补充 mob health/cooldown/reload 等隐藏
状态，并扩大输入范围或加入历史编码；latent distribution 本身不能弥补缺失的
Markov state。

## 验证

测试文件：`tests/test_stochastic_minigrid_env.py`、
`tests/test_minigrid_stochastic_latent.py`、
`tests/test_minigrid_stochastic_latent_v2.py`

运行：

```bash
python -m unittest tests/test_stochastic_minigrid_env.py
python -m unittest tests/test_minigrid_stochastic_latent.py
python -m unittest tests/test_minigrid_stochastic_latent_v2.py
```

环境测试覆盖：

- 默认 deterministic 行为；
- `p=0` 和 `p=1` 边界；
- 三个导航动作的 no-op failure；
- 交互动作保持确定性；
- seed 可复现；
- uniform/普通起点规则；
- 非法概率校验；
- 经验 failure rate；
- FullyObsWrapper 和 6-action compact contract。

WM latent 测试还覆盖旧 forward contract、v1 二分类 outcome、v2 prior/posterior
shape、latent sampling、标签监督和反向传播。

最近验证结果：环境测试 `9 passed`，v1 latent 测试 `4 passed`，v2 latent 测试
`4 passed`；相关 Python 文件通过 `py_compile` 和 `git diff --check`。

## 当前限制

当前代码已完成 `outcome_v1` 和 MiniGrid `latent_v2` 的训练与离线分布评估，尚未接入
stochastic imagined rollout。Crafter 的多实体随机行为、隐藏 health/cooldown 和
更大感受野需要后续 domain 接入。

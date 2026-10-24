# G1 SuperMimic 使用说明

当前 SuperMimic 使用本项目的 G1 29DoF 机器人与 mjlab，通过独立入口进行训练和评估。以下命令均在 `unitree_rl_mjlab` 项目根目录、已安装项目依赖的 Python 环境中执行。

## 1. 项目结构

```text
unitree_rl_mjlab/
├── scripts/
│   ├── csv_to_npz.py                 # 单个 CSV 转换，供批量脚本复用
│   ├── csv_to_npz_multi.py           # 批量 CSV → NPZ，可同时输出 MP4
│   ├── train_supermimic.py           # SuperMimic 训练与恢复入口
│   ├── evaluate_supermimic.py        # 单次、批量与持续评估入口
│   └── play_supermimic.py            # 查看器播放与策略 MP4 导出入口
├── src/assets/
│   ├── robots/unitree_g1/            # G1 模型、执行器和动作缩放
│   └── motions/g1/                   # CSV 与转换后的 NPZ 动作
├── src/tasks/supermimic/
│   ├── env.py                       # 物理步进、transition 捕获与局部 reset
│   ├── tracking_env_cfg.py           # 通用环境、奖励、终止和随机化配置
│   ├── config/
│   │   ├── __init__.py               # 环境创建工厂 make_env
│   │   └── g1/
│   │       ├── env_cfgs.py           # G1 训练与评估环境配置
│   │       └── rl_cfg.py             # SuperMimicConfig 与算法默认参数
│   ├── mdp/
│   │   ├── motion_lib.py             # 多 NPZ 加载、帧索引和 bin 划分
│   │   ├── commands.py               # 参考目标、重置初始化与动作选择
│   │   ├── sampling.py               # 基于回报的重置采样
│   │   ├── state.py                  # 状态表达和仿真状态读取
│   │   ├── rotations.py              # 可微旋转运算
│   │   ├── kinematics.py             # 可微正向运动学 FK
│   │   ├── functional.py             # 真实与想象轨迹共用的 MDP 计算
│   │   ├── observations.py           # 观测管理器适配
│   │   ├── rewards.py                # 奖励管理器适配
│   │   ├── terminations.py           # 失败、时间上限与动作结束判据
│   │   ├── logging_metrics.py        # 目标帧推进前的跟踪、奖励和终止快照
│   │   └── metrics.py                # 跟踪指标
│   ├── rl/
│   │   ├── networks.py               # 世界模型、策略和 Critic 网络
│   │   ├── buffer.py                 # 经验池、WM 窗口和想象起点采样
│   │   ├── algorithm.py              # WM 更新、策略 BPTT 和 Critic 更新
│   │   ├── runner.py                 # 采集、更新调度、日志与 checkpoint
│   │   ├── metrics.py                # 区间指标聚合与采样诊断
│   │   ├── logging.py                # CSV 与 W&B 写入、资源关闭
│   │   └── evaluation.py             # 逐动作评估与播放支持
│   └── doc/supermimic.md             # 本文档
└── logs/supermimic/<run>/            # 运行时生成
    ├── config.json                  # 本次训练配置
    ├── motions.json                 # 动作库清单
    ├── train_log.csv                # 训练指标
    ├── last.pt                      # 最近一次保存的 checkpoint
    ├── checkpoints/                 # 按迭代编号保存的 checkpoint
    └── evaluation/                  # 评估输出（默认位置）
```

算法和网络默认参数定义在 `src/tasks/supermimic/config/g1/rl_cfg.py`；机器人与评估环境配置定义在同目录的 `env_cfgs.py`。现有 PPO 入口 `scripts/train.py` 不用于 SuperMimic 训练。

## 2. 批量转换动作

将一个目录下的 G1 29DoF CSV 转换成同名 NPZ：

```bash
python scripts/csv_to_npz_multi.py \
  --robot g1 \
  --input-dir src/assets/motions/g1 \
  --output-dir src/assets/motions/g1 \
  --input-fps 30 \
  --output-fps 50
```

例如 `dance1_subject2.csv` 输出为 `dance1_subject2.npz`。请将 `--input-dir` 替换成自己的 CSV 目录，并按实际录制帧率设置 `--input-fps`。当前 SuperMimic 控制频率为 50 Hz，训练动作应使用 `--output-fps 50`。

同时为每个动作输出一份同名 MP4：

```bash
python scripts/csv_to_npz_multi.py \
  --robot g1 \
  --input-dir src/assets/motions/g1 \
  --output-dir src/assets/motions/g1 \
  --input-fps 30 \
  --output-fps 50 \
  --save-video True \
  --video-dir videos/reference_motions
```

这些视频展示参考动作转换过程，文件名为 `<CSV名称>.mp4`，相机跟踪 `robot`，视频帧率与 `--output-fps` 一致。

| 参数 | 用途 |
|---|---|
| `--input-dir` | CSV 输入目录，默认只扫描该目录顶层 |
| `--output-dir` | NPZ 输出目录；省略时为 `src/assets/motions/<robot>` |
| `--recursive` | 包含子目录，NPZ 和 MP4 均保留输入目录的相对结构 |
| `--save-video True` | 开启 MP4 输出；默认关闭，也可写作 `--save-video` |
| `--video-dir` | MP4 输出目录；省略时与 NPZ 输出目录相同 |
| `--overwrite` | 重新转换已有输出；默认跳过已有 NPZ，以及开启视频时已有的对应 MP4 |
| `--device cpu` | 指定 CPU；默认 `cuda:0`，CUDA 不可用时自动回退到 CPU |
| `--line-range 1 300` | 每个 CSV 只读取第 1～300 行，包含两端 |

脚本按文件顺序逐个转换，某个文件失败后仍继续处理其余文件，最后汇总成功、跳过和失败数量；存在失败文件时返回非零退出码。开启视频后，如果只有 NPZ、缺少 MP4，该动作会重新转换。

训练和评估的 `--motion-file` 接受单个 NPZ 或目录。目录按文件名排序读取顶层 `*.npz`，**不递归读取子目录**。使用 `--recursive` 转换后，应将 `--motion-file` 指向实际包含所需 NPZ 的目录；若希望一次训练全部动作，需要把这组 NPZ 放在同一目录中。恢复和评估时保持与 checkpoint 对应的动作集合及顺序一致。

## 3. 训练与恢复

### 开始训练

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --log-dir logs/supermimic/g1 \
  --device cuda:0
```

训练默认同时记录本地 CSV 和在线 W&B，请先配置 W&B 登录；不需要在线记录时添加 `--logger csv`，或使用 `--wandb-mode offline` 离线记录。配置与动作清单仍保存在本地 JSON。`--motion-file` 也可以直接指定一个 NPZ。常用参数如下，未传入时采用 `config/g1/rl_cfg.py` 的默认值：

| 参数 | 默认值 | 用途 |
|---|---:|---|
| `--num-envs` | 4096 | 并行环境数量 |
| `--iterations` | 320000 | 总训练迭代数 |
| `--buffer-steps` | 1000 | 每个环境保留的缓存步数 |
| `--utd` | 0.4 | 每次迭代平均采集 `1 / utd` 个环境步，默认按 2/3 步交替采集 |
| `--wm-window` | 8 | 世界模型自回归训练窗口长度 |
| `--wm-batches` / `--wm-batch` | 10 / 2048 | 每次迭代 WM 更新批数 / 每批窗口数 |
| `--pi-batch` / `--horizon` | 4096 / 32 | 想象轨迹数量 / 每条轨迹长度 |
| `--critic-epochs` / `--critic-minibatches` | 5 / 4 | 每轮 Critic 更新的 epoch 数 / minibatch 数 |
| `--lr` | 0.0002 | 初始学习率 |
| `--gamma` / `--lam` | 0.99 / 0.95 | 折扣系数 / λ-return 系数 |
| `--action-noise` | 0.1 | 真实采集时动作探索噪声的标准差 |
| `--log-every` / `--ckpt-every` | 50 / 2000 | 日志记录 / checkpoint 保存间隔，单位为迭代 |
| `--amp none` | 默认 `bf16` | 关闭网络 BF16 混合精度 |
| `--no-compile` | 默认开启编译 | 关闭 `torch.compile` |

例如，调整并行数量与 batch 大小：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --log-dir logs/supermimic/g1_1024 \
  --num-envs 1024 \
  --wm-batch 1024 \
  --pi-batch 1024
```

### 日志与 W&B

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--logger` | `wandb` | `wandb` 同时写 CSV 和 W&B；`csv` 只写本地 CSV |
| `--wandb-project` | `unitree-supermimic` | W&B 项目名 |
| `--wandb-entity` | 当前账户默认设置 | W&B 账户或团队 |
| `--wandb-name` | 运行目录名和启动时间 | 本次 W&B run 的显示名称 |
| `--wandb-mode` | `online` | `offline` 将 W&B 记录保存在本地，之后可同步 |

例如，仅使用本地 CSV：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --log-dir logs/supermimic/g1_local \
  --logger csv
```

CSV 和 W&B 每 `--log-every` 次迭代记录同一组聚合结果，W&B 横轴使用 `iteration`。在线初始化失败会明确报错，可改用 `--logger csv` 或 `--wandb-mode offline`。本轮只记录标量和运行配置，不自动上传 checkpoint、视频或梯度直方图。

`train_log.csv` 保留原有 17 列：`iteration`、`wall_h`、`sim_steps`、`env_steps`、`lr`、`wm_loss`、`wm_err_last`、`pi_loss`、`v_loss`、`pi_gradnorm`、`imag_survival`、`imag_reward`、`policy_skipped`、`sim_reward`、`episode_length`、`episode_return`、`failure_fraction`。其中 `episode_return` 仍是按实际步数和最大单步奖励归一化后的回报，`episode_length` 的单位为控制步；新增原始回报单独记录。

| 新增指标 | 含义与统计口径 |
|---|---|
| `Tracking/error_anchor_pos`、`error_anchor_rot`、`error_anchor_lin_vel`、`error_anchor_ang_vel` | 锚点位置、姿态、线速度、角速度误差；按记录区间的环境控制步求平均 |
| `Tracking/error_body_pos`、`error_body_rot`、`error_body_lin_vel`、`error_body_ang_vel` | 跟踪身体对应误差；先在身体间求平均，再按环境控制步平均 |
| `Tracking/error_joint_pos`、`error_joint_vel` | 整条关节位置或速度误差向量的 L2 范数，非逐关节 RMSE |
| `Reward/<奖励项>` | 当前 8 项奖励乘权重和控制时间步后的贡献，按环境控制步平均；各项之和等于原始平均单步奖励 |
| `Termination/<原因>_count`、`<原因>_fraction` | `time_out`、`motion_end`、`anchor_pos`、`anchor_ori` 的结束次数，以及占区间内结束 episode 的比例 |
| `Episode/completed`、`return_raw`、`length_seconds` | 结束回合数、原始累计回报均值和实际回合时长均值；按结束的回合数加权 |
| `Sampling/entropy`、`top1_prob`、`top1_bin` | 当前 bin 采样分布的归一化熵、最大概率、对应整数 bin ID |
| `Performance/collection_s`、`wm_update_s`、`policy_critic_update_s` | 每迭代的数据采集、WM 更新、策略和 Critic 更新平均壁钟耗时，单位为秒 |
| `Performance/iterations_per_s`、`transitions_per_s` | 记录区间每秒迭代数和真实 transition 数；一轮并行环境步产生 `num_envs` 条 transition |

位置、姿态、线速度、角速度误差分别使用米、弧度、米/秒、弧度/秒。身体位置和姿态沿用当前参考对齐，锚点保留 `pelvis`。所有跟踪误差在动作执行后、目标帧推进前计算，与本步奖励使用同一个目标帧 `f`。

同一回合可能同时触发多个终止原因，所以原因比例之和允许超过 100%。区间内没有结束回合时，次数记零，回合均值和终止比例留空。采样熵除以 `log(num_bins)`，只有一个 bin 时为 1；查询日志不更新采样器历史。性能计时不逐步强制 GPU 同步，反映整体执行开销，不代表单独的 GPU 内核耗时。

首次日志的跟踪、奖励和回合统计包含经验池预填充期间的真实采集；`Performance/` 指标从预填充结束后的训练主循环开始统计，不包含预填充耗时和步数。

### 预热开关

默认开启 `--enable-warmup`：先采集 64 步填充经验池，前 500 次迭代只更新世界模型，之后再进行策略和 Critic 更新。可通过 `--warmup-steps` 和 `--wm-only-iters` 调整。

关闭预热：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --log-dir logs/supermimic/g1_no_warmup \
  --no-enable-warmup
```

关闭后，从开始就按默认的 2/3 步交替采集；缓存满足 WM 窗口长度时先更新 WM，再尝试更新策略和 Critic。没有满足想象长度要求的起点时，跳过策略和 Critic 更新。

### 恢复训练

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --log-dir logs/supermimic/g1 \
  --resume logs/supermimic/g1/last.pt \
  --iterations 400000
```

`--iterations 400000` 表示训练到总计 400000 次迭代，而不是额外增加 400000 次。

恢复时读取 checkpoint 中的训练配置、网络、优化器、归一化统计、bin 采样统计和训练进度。经验池与物理仿真现场不保存，需要重新采集：开启预热时先填充至少 `max(warmup_steps, wm_window)` 步；关闭预热时按普通采集节奏逐步补充。已经完成的 WM 专属预热迭代不会重新开始。

恢复时可覆盖总迭代数、环境数量、缓存容量、日志/保存周期、初始填充步数、预热开关、编译/精度配置，以及上表中的全部 W&B/日志选项；其余算法与网络配置应保持 checkpoint 原值。省略参数时沿用 checkpoint 的设置，旧 checkpoint 没有的新日志选项使用当前默认值。

每次启动训练（包括恢复）都创建新的 W&B run，并记录恢复来源 checkpoint 和起始迭代；相同 `--wandb-name` 只影响显示名，不复用 run ID。训练迭代从断点继续，CSV 仍追加到指定 `--log-dir`。旧 CSV 会先扩展表头，历史行保留原值，新增指标列留空。

## 4. 评估

### 评估单个 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --checkpoint-file logs/supermimic/g1/last.pt \
  --output-dir logs/supermimic/g1/evaluation \
  --device cuda:0
```

评估为每个动作创建一个环境，从首帧初始化，运行到动作结束或失败，每个动作只计分一次。关闭训练时的观测/动作噪声和重置扰动，编码器偏差与质心偏移归零，机器人和地面摩擦设为 1.0，并取消训练时的 10 秒 episode 上限。

结果包括：

- `<checkpoint名称>.json`：逐动作回报、归一化回报、执行步数及失败情况，以及整体汇总。
- `eval.csv`：各 checkpoint 的迭代数、平均归一化回报 `norm_return`、按动作长度加权的回报 `norm_return_framew`、失败数量和失败动作列表。

每个动作的归一化分母为完整动作的 `(帧数 − 1) × 单步奖励上界`，提前失败不会缩短分母。

省略 `--output-dir` 时，单 checkpoint 评估输出到该 checkpoint 所在目录的 `evaluation/`。若需要同时保存各动作的机器人状态轨迹，在命令后添加 `--record`；轨迹保存为输出目录下 `<checkpoint名称>/<动作名称>.npz`，包含 `states`、`clip_start` 和 `fps`。这不是 MP4，也不是训练输入参考动作的七字段 NPZ 格式。

### 评估一个运行目录中的 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --run-dir logs/supermimic/g1
```

读取该运行目录的 `checkpoints/ckpt_*.pt`，跳过输出目录中已有对应 JSON 的 checkpoint；默认输出到运行目录的 `evaluation/`。

### 持续评估新 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1 \
  --run-dir logs/supermimic/g1 \
  --watch
```

`--watch` 必须配合 `--run-dir`，每轮处理完现有 checkpoint 后等待 30 秒，再扫描新增文件。使用 `Ctrl+C` 结束。`--checkpoint-file` 与 `--run-dir` 二选一，`--record` 和 `--output-dir` 可用于上述各评估方式。

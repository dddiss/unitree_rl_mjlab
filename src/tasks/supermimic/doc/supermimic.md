# G1 SuperMimic 使用说明

当前 SuperMimic 使用本项目的 G1 29DoF 机器人与 mjlab，通过独立入口进行训练、评估和播放。以下命令均在 `unitree_rl_mjlab` 项目根目录、已安装项目依赖的 Python 环境中执行。

使用顺序：准备动作 CSV → 转成 50 Hz NPZ → 训练或恢复 → 评估 → 查看器播放或导出策略视频。已有 LAFAN 格式 CSV 可从第 2.3 节开始；已有兼容 NPZ 可从第 2.4 节开始。

## 1. 项目结构

```text
unitree_rl_mjlab/
├── scripts/
│   ├── convert_soma_csv_to_lafan.py  # SOMA/Bones G1 CSV → LAFAN 格式 CSV
│   ├── filter_and_copy_soma_csv.py   # 按 SONIC 关键词规则筛选并复制 CSV
│   ├── csv_to_npz.py                 # 单个 CSV 转换，供批量脚本复用
│   ├── csv_to_npz_multi.py           # 批量 CSV → NPZ，可同时输出 MP4
│   ├── train_supermimic.py           # SuperMimic 训练与恢复入口
│   ├── evaluate_supermimic.py        # 单次、批量与持续评估入口
│   └── play_supermimic.py            # 查看器播放与策略 MP4 导出入口
├── src/assets/
│   ├── robots/unitree_g1/            # G1 模型、执行器和动作缩放
│   └── motions/
│       ├── g1/                       # LAFAN 格式 CSV 动作
│       └── g1_npz/                   # 本文示例的 NPZ 输出根目录，可含子目录
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

算法和网络默认参数定义在 `src/tasks/supermimic/config/g1/rl_cfg.py`；机器人与评估环境配置定义在同目录的 `env_cfgs.py`。SuperMimic 的播放入口为 `scripts/play_supermimic.py`；原有 PPO tracking 使用 `scripts/train.py` 和 `scripts/play.py`，两套入口的 checkpoint 格式不同。

## 2. 动作准备与加载

### 2.1 SOMA/Bones-SEED G1 CSV 转为 LAFAN 格式

输入是每个动作一份带表头的 CSV，包含 `root_translateX/Y/Z`、`root_rotateX/Y/Z` 和 29 个 `*_joint_dof` 列。位置单位为厘米，根欧拉角和关节角单位为度。这与部署用的 `joint_pos.csv`、`body_pos.csv` 等拆分文件不同，不能将后者直接作为此脚本输入。

```bash
python scripts/convert_soma_csv_to_lafan.py \
  --input /path/to/soma/g1/csv \
  --output data/bones_lafan \
  --fps-source 120 \
  --fps 30 \
  --num-workers 16
```

请按原始数据设置 `--fps-source`。上例每隔 4 帧取一帧，将 120 Hz 转为 30 Hz；脚本支持保持帧率或整数倍降采样，不支持直接按此方式将 120 Hz 转为 50 Hz。后续由 NPZ 转换脚本插值到 50 Hz。

输出为无表头的 36 列 CSV，每行依次为：

```text
根位置 x、y、z（米）
+ 根四元数 qx、qy、qz、qw
+ 29 个关节角度（弧度，LAFAN G1 顺序）
```

关节按列名读取并重排，即使输入列顺序不同也保持相同输出顺序。该步骤只转换单位、旋转表示和帧率，不重新进行动作重定向或 FK，也不输出速度。

目录会自动递归扫描，保留所有相对子目录和动作文件名；例如 `session_a/walk.csv` 仍输出到 `session_a/walk.csv`。默认跳过已有输出，添加 `--overwrite` 可重新转换。输入、输出目录需相互独立，不能相同或互相嵌套。

### 2.2 按动作名称筛选 CSV（可选）

```bash
python scripts/filter_and_copy_soma_csv.py \
  --source data/bones_lafan \
  --dest data/bones_lafan_filtered \
  --workers 16
```

默认沿用 SONIC 的 Bones 关键词规则，例如排除名称包含 `chair`、`bike`、`climb`、`stair`、`handstand`、`box_jump` 的动作。匹配对象是文件名及其直接父目录名称，不区分大小写；只按名称筛选，不读取动作内容，也不检查碰撞或物理可行性。

| 参数 | 用途 |
|---|---|
| `--source` / `--dest` | 输入和筛选后的 CSV 根目录，需相互独立、不嵌套 |
| `--workers 16` | 并行复制数量 |
| `--dry-run --verbose` | 预览通过和排除的文件，不写入输出 |
| `--filter-keywords chair bike` | 用指定关键词替换默认排除列表；不跟关键词可关闭排除规则 |
| `--add-keywords crawl` | 在默认或指定的排除列表中追加关键词 |
| `--filter-file keep.txt` | 每行一个保留关键词，仍优先应用排除规则 |
| `--overwrite` | 覆盖已有输出；默认跳过 |

筛选脚本也递归处理并保留完整相对路径，复制后的 CSV 内容不变，原始文件保留。更改筛选规则时建议使用新的目标目录：脚本不会删除旧目录中先前已复制的文件。也可以先筛选原始 SOMA CSV，再转换通过筛选的文件。

### 2.3 LAFAN 格式 CSV 批量转为 NPZ

将一个目录下的 G1 29DoF CSV 转换成同名 NPZ。以下示例将参考 CSV 和生成的 NPZ 分目录存放，已有其他输出目录也可继续使用：

```bash
python scripts/csv_to_npz_multi.py \
  --robot g1 \
  --input-dir src/assets/motions/g1 \
  --output-dir src/assets/motions/g1_npz \
  --input-fps 30 \
  --output-fps 50 \
  --recursive
```

例如 `dance1_subject2.csv` 输出为 `dance1_subject2.npz`。请将 `--input-dir` 替换成自己的 CSV 目录，并按实际录制帧率设置 `--input-fps`。当前 SuperMimic 控制频率为 50 Hz，训练动作应使用 `--output-fps 50`。

对于前两步得到的 SOMA/Bones 数据，可输出到同一个 NPZ 根目录的独立子目录：

```bash
python scripts/csv_to_npz_multi.py \
  --robot g1 \
  --input-dir data/bones_lafan_filtered \
  --output-dir src/assets/motions/g1_npz/bones_seed \
  --input-fps 30 \
  --output-fps 50 \
  --recursive
```

CSV 文件本身不记录帧率，因此这里的 `--input-fps` 必须与上一阶段的 `--fps` 一致。若跳过筛选，改用 `data/bones_lafan` 作为输入目录。

同时为每个动作输出一份同名 MP4：

```bash
python scripts/csv_to_npz_multi.py \
  --robot g1 \
  --input-dir src/assets/motions/g1 \
  --output-dir src/assets/motions/g1_npz \
  --input-fps 30 \
  --output-fps 50 \
  --recursive \
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

### 2.4 训练、评估与播放如何加载动作

三种入口的 `--motion-file` 都接受单个 NPZ 或目录。指定目录时会自动递归读取全部子目录中的 `*.npz`，按路径排序确定动作顺序，无需额外传入 `--recursive`。注意：`--recursive` 是 CSV 批量转换脚本的开关，不能传给训练、评估或播放入口。

本示例统一使用 `--motion-file src/assets/motions/g1_npz`，它会同时加载根目录与 `bones_seed/` 等子目录中的动作。只想用某个子目录训练时，可直接指定那个子目录。

恢复、评估和播放会核对 checkpoint 的动作清单，需保持动作集合、顺序和帧数一致。多动作 checkpoint 只播放其中一个动作时，仍加载原来的整套动作库，通过 `--clip` 选择，不能把 `--motion-file` 改成单个动作来绕过清单检查。

当前动作显示名取文件名、不包含子目录。不同目录中的同名文件可以加载，但 `--clip` 按名称选择时只匹配第一个，评估轨迹和视频也使用该名称保存。准备数据时应保证文件名在整套动作库中唯一，避免选择歧义及导出覆盖。

## 3. 训练、恢复与日志

### 3.1 开始训练

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --log-dir logs/supermimic \
  --run-name g1 \
  --device cuda:0
```

训练默认同时记录本地 CSV 和在线 W&B，请先配置 W&B 登录；不需要在线记录时添加 `--logger csv`，或使用 `--wandb-mode offline` 离线记录。配置与动作清单仍保存在本地 JSON。`--motion-file` 也可以直接指定一个 NPZ。常用参数如下，未传入时采用 `config/g1/rl_cfg.py` 的默认值：

`rl_cfg.py` 中的 `log_dir="logs/supermimic"` 指定日志根目录，`run_name="g1"` 指定基础实验名称。每次启动生成一次 `YYYYMMDD_HHMMSS_微秒` 时间戳，最终目录为 `log_dir/run_name_时间戳`，W&B 显示名称与这个带时间戳的目录名完全一致。可通过 `--log-dir` 和 `--run-name` 分别覆盖；旧参数 `--wandb-name` 保留为 `--run-name` 的别名。`run_name` 必须为单个文件夹名称。

例如，实际保存目录为 `logs/supermimic/g1_20261024_180000_123456/`，W&B 名称就是 `g1_20261024_180000_123456`。下文恢复、评估和播放命令中的这个示例目录需要替换为你的实际目录。

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

动作探索的环境比例直接在 `src/tasks/supermimic/config/g1/rl_cfg.py` 中配置，不新增命令行参数：

```python
action_noise: float = 0.1      # 有探索噪声环境的高斯噪声标准差
collect_det_frac: float = 0.25 # 固定一部分环境关闭动作探索噪声
```

前 `round(collect_det_frac × num_envs)` 个环境始终不添加动作探索噪声，reset 后分组不变。默认 4096 个环境中，编号 0～1023 不加动作噪声，其余 3072 个环境每步添加标准差 0.1 的独立高斯噪声。比例设为 `0.0` 恢复全部加噪，设为 `1.0` 则全部关闭动作探索噪声。观测噪声、物理随机化、动作缩放和加噪后的 `[-20, 20]` 裁剪保持原有设置；经验池和历史观测仍使用实际执行的动作。

新训练与恢复训练都使用当前 `rl_cfg.py` 中的 `collect_det_frac`，实际取值会保存到本次配置和 checkpoint，旧 checkpoint 缺少该字段也可恢复。`action_noise` 的原有恢复规则保持不变：新训练采用本地配置，恢复训练沿用 checkpoint 中保存的强度。该分组同样作用于初始经验池填充和 WM 预热的数据采集；不改变想象训练及评估的噪声设置。

例如，调整并行数量与 batch 大小：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --log-dir logs/supermimic \
  --run-name g1_1024 \
  --num-envs 1024 \
  --wm-batch 1024 \
  --pi-batch 1024
```

### 3.2 预热开关

初始经验池填充始终执行，默认先采集 64 个环境步；`--warmup-steps` 可调整填充步数，并至少满足 `wm_window`。`--enable-warmup` 只控制前 500 次迭代的 WM 专属预热，长度由 `--wm-only-iters` 调整。

预热期间允许抽到跨 episode 的 8 步缓存窗口，保留原来的终止掩码：终止那一步参与损失，后面的跨回合数据不参与该窗口的自回归损失。预热结束后，必须采到同一 episode 内完整的 8 条 transition，允许第 8 条终止或截断；没有合法窗口时继续采集，跳过当轮 WM、策略和 Critic 更新。

关闭预热：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --log-dir logs/supermimic \
  --run-name g1_no_warmup \
  --no-enable-warmup
```

关闭后，仍先填充默认 64 步，然后按默认的 2/3 步交替采集，直接使用严格的同回合 WM 窗口。满足条件时先更新 WM，再尝试更新策略和 Critic。没有满足想象长度要求的起点时，仍只跳过策略和 Critic 更新。

回报采样会在 episode 结束时，仅更新初始化帧 `k` 所属的起点 bin。分数采用 allmimic 的有限长度归一化方式：实际折扣回报除以 `r_max × (1 - gamma^H) / (1 - gamma)`，其中 `H` 是从 `k` 到动作末帧的有效控制步数；启用 `time_out` 时，再取它与最大回合步数的较小值。删除 `time_out` 后，分母只按动作剩余步数计算。提前失败不会缩短分母，失败、超时和动作结束均不为这项采样统计添加自举尾项。

同批多个 episode 从同一 bin 开始时，先对各自归一化后的分数求平均，再更新一次 EMA。回报 EMA 的新分数权重在 `rl_cfg.py` 中通过 `bin_ema: float = 0.1` 设置，默认与 allmimic 相同，更新公式为 `(1 - bin_ema) × 旧值 + bin_ema × 新值`。取值范围为 `[0, 1]`：`0` 不更新分数，`1` 直接使用本批平均分数；首次仍从零更新，不额外截断分数。新训练和恢复训练均读取当前 `rl_cfg.py` 的 `bin_ema`，不新增命令行参数，实际值写入配置与 checkpoint；旧 checkpoint 缺少该字段也可恢复。

采样仍混合 50% 均匀、40% 当前 softmax 和 10% 历史分布；历史分布的 `0.999 × 旧值 + 0.001 × 当前分布` 不受 `bin_ema` 影响。未引入 allmimic 的概率上限等额外设置。`reset k → 目标 k+1` 的时序、经验池和策略/Critic 的自举逻辑不变。

### 3.3 恢复训练

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --log-dir logs/supermimic \
  --resume logs/supermimic/g1_20261024_180000_123456/last.pt \
  --iterations 400000
```

`--iterations 400000` 表示训练到总计 400000 次迭代，而不是额外增加 400000 次。

恢复时读取 checkpoint 中的训练配置、网络、优化器、归一化统计、bin 采样统计和训练进度。经验池与物理仿真现场不保存，因此无论预热开关是否开启，都先重新填充 `max(warmup_steps, wm_window)` 步。已经完成的 WM 专属预热迭代不会重新开始，是否使用严格窗口由恢复后的迭代数和预热开关决定。

新 checkpoint 保存有限长度归一化的起点 bin 分数。此前包含自举、沿途 bin 回报的旧统计无法准确转换；恢复这类 checkpoint 时会明确提示并重置 bin 分数和历史采样分布，再从新回合积累，网络、优化器和训练进度仍正常恢复。原 checkpoint 文件不改写。

当前 Critic 已移除独立重复的 6 维根相对朝向，并加入 29 维上一动作（无历史）；未来参考目标中的朝向保留，默认 Critic 输入为 847 维。此前 824 维 Critic 的 checkpoint 不能直接完整恢复训练；Actor 结构未变，仍可用于评估和播放。

恢复时可覆盖总迭代数、环境数量、缓存容量、日志/保存周期、初始填充步数、预热开关、编译/精度配置，以及第 3.4 节中的全部 W&B/日志选项；`collect_det_frac` 和 `bin_ema` 始终读取当前 `rl_cfg.py`。其余算法与网络配置应保持 checkpoint 原值。省略参数时沿用 checkpoint 的设置，旧 checkpoint 没有的新日志选项使用当前默认值。

每次启动训练（包括恢复）都生成新的带时间戳的实验目录和独立 W&B run，CSV 写入新目录，训练迭代从断点继续，并记录恢复来源 checkpoint 和起始迭代。配置和 checkpoint 保留日志根目录及基础名称，不保存自动追加的后缀，因此恢复时不会叠加旧时间戳。旧 checkpoint 的非空 `wandb_name` 会映射为基础 `run_name`；旧 `log_dir` 按保存值作为新日志根目录，命令行仍可覆盖。

### 3.4 日志、W&B 与终端输出

| 参数 | 默认值 | 用途 |
|---|---|---|
| `--logger` | `wandb` | `wandb` 同时写 CSV 和 W&B；`csv` 只写本地 CSV |
| `--wandb-project` | `unitree-supermimic` | W&B 项目名 |
| `--wandb-entity` | 当前账户默认设置 | W&B 账户或团队 |
| `--log-dir` | `logs/supermimic` | 日志根目录，可在 `rl_cfg.py` 中设置 |
| `--run-name` | `g1` | 基础实验名称，追加一次时间戳后同时用于实验目录和 W&B 名称 |
| `--wandb-mode` | `online` | `offline` 将 W&B 记录保存在本地，之后可同步 |

例如，仅使用本地 CSV：

```bash
python scripts/train_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --log-dir logs/supermimic \
  --run-name g1_local \
  --logger csv
```

CSV 和 W&B 每 `--log-every` 次迭代记录同一组聚合结果，W&B 横轴使用 `iteration`。在线初始化失败会明确报错，可改用 `--logger csv` 或 `--wandb-mode offline`。当前只记录标量和运行配置，不自动上传 checkpoint、视频或梯度直方图。

`train_log.csv` 保留原有 17 列：`iteration`、`wall_h`、`sim_steps`、`env_steps`、`lr`、`wm_loss`、`wm_err_last`、`pi_loss`、`v_loss`、`pi_gradnorm`、`imag_survival`、`imag_reward`、`policy_skipped`、`sim_reward`、`episode_length`、`episode_return`、`failure_fraction`。其中 `episode_return` 仍是按实际步数和最大单步奖励归一化后的回报，`episode_length` 的单位为控制步；新增原始回报单独记录。

W&B 将网络损失统一归入 `Loss/`：`Loss/wm_loss`、`Loss/wm_err_last`（WM 窗口末步误差）、`Loss/pi_loss`、`Loss/v_loss`。CSV 仍使用上述原始列名，两端数值和区间平均方式相同；没有更新时留空，不上传对应 W&B 指标。

| 新增指标 | 含义与统计口径 |
|---|---|
| `Tracking/error_anchor_pos`、`error_anchor_rot`、`error_anchor_lin_vel`、`error_anchor_ang_vel` | 锚点位置、姿态、线速度、角速度误差；按记录区间的环境控制步求平均 |
| `Tracking/error_body_pos`、`error_body_rot`、`error_body_lin_vel`、`error_body_ang_vel` | 跟踪身体对应误差；先在身体间求平均，再按环境控制步平均 |
| `Tracking/error_joint_pos`、`error_joint_vel` | 整条关节位置或速度误差向量的 L2 范数，非逐关节 RMSE |
| `Reward/<奖励项>` | 当前 8 项奖励贡献按环境控制步平均后除以 `step_dt`，跟踪奖励的显示上限为对应权重；各项之和乘以 `step_dt` 等于原始平均单步奖励，实际训练奖励不变 |
| `Termination/<原因>_fraction` | `time_out`、`motion_end`、`anchor_pos`、`anchor_ori` 各原因占区间内结束 episode 的比例，不记录各原因的次数 |
| `Episode/completed`、`return_raw`、`length_seconds` | 结束回合数、原始累计回报均值和实际回合时长均值；按结束的回合数加权 |
| `Sampling/entropy`、`top1_prob`、`top1_bin` | 当前 bin 采样分布的归一化熵、最大概率、对应整数 bin ID |
| `Performance/collection_s`、`wm_update_s`、`policy_critic_update_s` | 每迭代的数据采集、WM 更新、策略和 Critic 更新平均壁钟耗时，单位为秒 |
| `Performance/iterations_per_s`、`transitions_per_s` | 记录区间每秒迭代数和真实 transition 数；一轮并行环境步产生 `num_envs` 条 transition |

位置、姿态、线速度、角速度误差分别使用米、弧度、米/秒、弧度/秒。身体位置和姿态沿用当前参考对齐，锚点保留 `pelvis`。所有跟踪误差在动作执行后、目标帧推进前计算，与本步奖励使用同一个目标帧 `f`。

同一回合可能同时触发多个终止原因，所以原因比例之和允许超过 100%。区间内没有结束回合时，`Episode/completed` 为零，回合均值和终止比例留空。采样熵除以 `log(num_bins)`，只有一个 bin 时为 1；查询日志不更新采样器历史。性能计时不逐步强制 GPU 同步，反映整体执行开销，不代表单独的 GPU 内核耗时。

首次日志的跟踪、奖励和回合统计包含经验池预填充期间的真实采集；`Performance/` 指标从预填充结束后的训练主循环开始统计，不包含预填充耗时和步数。

终端同样按 `--log-every` 打印，默认每 50 次迭代输出一次 mjlab 风格的摘要；训练正常结束时若还有未满一个区间的记录，也会输出剩余区间。终端包括：

- 当前迭代、吞吐、区间耗时、平均每迭代耗时，以及采集、WM、策略和 Critic 更新阶段的平均耗时。
- `Loss/wm_loss`、`Loss/wm_err_last`、`Loss/pi_loss`、`Loss/v_loss` 和 `Training/lr`。
- 紧接学习率下方的 `Episode/length_steps`，沿用 CSV 的 `episode_length` 统计，单位为控制步。
- 各项 `Reward/` 和 `Termination/*_fraction`，以及累计 transition 数、总运行时间和预计剩余时间。

尚未产生有效统计的项目显示 `--`，WM 专属预热期间不估算 ETA。终端使用同一份聚合结果，但只展示上述项目；CSV/W&B 仍保留更完整的指标，包括策略梯度、想象奖励与存活率等。

## 4. 评估

### 4.1 评估单个 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --checkpoint-file logs/supermimic/g1_20261024_180000_123456/last.pt \
  --output-dir logs/supermimic/g1_20261024_180000_123456/evaluation \
  --device cuda:0
```

评估为每个动作创建一个环境，从首帧初始化，运行到动作结束或失败，每个动作只计分一次。默认关闭训练时的观测/动作噪声和重置扰动，编码器偏差与质心偏移归零，机器人和地面摩擦设为 1.0，并取消训练时的 10 秒 episode 上限。可通过第 4.4 节的参数单独开启噪声对比。

结果包括：

- `<checkpoint名称>.json`：逐动作回报、归一化回报、执行步数及失败情况，以及整体汇总。
- `eval.csv`：各 checkpoint 的迭代数、平均归一化回报 `norm_return`、按动作长度加权的回报 `norm_return_framew`、失败数量和失败动作列表。

每个动作的归一化分母为完整动作的 `(帧数 − 1) × 单步奖励上界`，提前失败不会缩短分母。

省略 `--output-dir` 时，单 checkpoint 评估输出到该 checkpoint 所在目录的 `evaluation/`。若需要同时保存各动作的机器人状态轨迹，在命令后添加 `--record`；轨迹保存为输出目录下 `<checkpoint名称>/<动作名称>.npz`，包含 `states`、`clip_start` 和 `fps`。这不是 MP4，也不是训练输入参考动作的七字段 NPZ 格式。

### 4.2 评估一个运行目录中的 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --run-dir logs/supermimic/g1_20261024_180000_123456
```

读取该运行目录的 `checkpoints/ckpt_*.pt`，跳过输出目录中已有对应 JSON 的 checkpoint；默认输出到运行目录的 `evaluation/`。

### 4.3 持续评估新 checkpoint

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --run-dir logs/supermimic/g1_20261024_180000_123456 \
  --watch
```

`--watch` 必须配合 `--run-dir`，每轮处理完现有 checkpoint 后等待 30 秒，再扫描新增文件。使用 `Ctrl+C` 结束。`--checkpoint-file` 与 `--run-dir` 二选一，`--record` 和 `--output-dir` 可用于上述各评估方式。

### 4.4 同一 checkpoint 的加噪对比评估

对比确定性评估与带噪运行，使用同一个 checkpoint 依次运行四种模式、每种模式重复多个随机种子：

```bash
python scripts/evaluate_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --checkpoint-file logs/supermimic/g1_20261024_180000_123456/checkpoints/ckpt_064000.pt \
  --noise-mode all \
  --seeds 0 1 2 \
  --output-dir logs/supermimic/noise_comparison \
  --device cuda:0
```

| `--noise-mode` | 观测噪声 | 动作探索噪声 |
|---|---|---|
| `none` | 关闭 | 关闭 |
| `action` | 关闭 | 开启 |
| `observation` | 开启 | 关闭 |
| `both` | 开启 | 开启 |
| `all` | 依次运行上述四组 | 依次运行上述四组 |

观测噪声复用训练采集的定义：本体观测噪声，以及未来参考目标中 6D 朝向的 `U(-0.05, 0.05)` 噪声，均在 RunningNorm 之前添加；评估不更新归一化统计或网络参数。动作噪声逐控制步、逐维度采样高斯噪声，在 `action_scale` 缩放之前添加，标准差默认读取 checkpoint 中的 `action_noise`（通常为 0.1），可用 `--action-noise 0.1` 覆盖。

各组仍从动作首帧初始化，目标为下一帧，保持相同的固定物理参数、无重置扰动和计分口径。因此这只是噪声对比，并非恢复完整训练环境。`--seeds` 控制每次重复的随机种子；进入对比模式后省略该参数则使用 `0 1 2`。仅评估某一组时，例如使用 `--noise-mode action --seeds 0 1 2`。

默认保持原评估的**不裁剪动作**行为，避免引入第二个变量。若要同时验证训练采集的 `[-20, 20]` 限幅，可在上述命令中添加 `--action-clip 20`，限幅在加噪后进行，并统一应用于四组。更改噪声强度或裁剪设置时，使用新的 `--output-dir` 保存对比结果。

对比输出与原有 `eval.csv` 分开，写入 `<输出目录>/noise_evaluation/<checkpoint名称>/`：

- `eval.csv`：每种模式、每个种子的回报和失败动作列表，以及实际噪声/裁剪设置。
- `summary.csv` / `summary.json`：各模式跨种子的回报、失败数量的均值与总体标准差；单个种子时标准差为 0。
- `<模式>/seed_<种子>.json`：每个动作的回报、有效步数和失败情况。
- 添加 `--record` 时，轨迹保存到 `<模式>/seed_<种子>/<动作名称>.npz`。

四组、三个种子共运行 12 次全动作评估。汇总表只包含本次命令选择的模式和种子，重新执行不会在该表重复追加同一个种子。也支持 `--run-dir` / `--watch`；已存在且设置相同的逐次报告会复用，设置不同则要求换输出目录。未传入上述新增参数时，继续使用原有确定性评估流程及输出路径。

## 5. 播放与视频导出

播放入口为 `scripts/play_supermimic.py`。它读取本地 SuperMimic checkpoint，使用评估环境和确定性策略，关闭训练时的观测/动作噪声及随机扰动。下列运行目录需要替换为你的实际 checkpoint 目录，动作目录需要与该 checkpoint 匹配。

### 5.1 使用 Viser 或 Native 查看器

```bash
python scripts/play_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --checkpoint-file logs/supermimic/g1_20261024_180000_123456/last.pt \
  --viewer viser \
  --clip dance1_subject2 \
  --device cuda:0
```

Viser 启动后，在浏览器打开终端提示的地址。改为 `--viewer native` 可使用本地 MuJoCo 查看器，需有可用的图形显示环境。

| 参数 | 用途 |
|---|---|
| `--motion-file` | 必填，原训练动作库的单个 NPZ 或根目录；目录自动递归加载 |
| `--checkpoint-file` | 必填，本地 SuperMimic checkpoint，例如 `last.pt` 或 `checkpoints/ckpt_002000.pt` |
| `--clip dance1_subject2` | 动作文件名，不带 `.npz`；只选择初始播放动作，不减少动作库加载范围 |
| `--viewer viser` | 浏览器查看器，可暂停后切换动作和初始化帧 |
| `--viewer native` | 本地 MuJoCo 查看器 |
| `--viewer auto` | 默认值；macOS 或检测到 `DISPLAY`/`WAYLAND_DISPLAY` 时使用 Native，否则使用 Viser |
| `--video-dir` | 改为导出 MP4，详见第 5.3 节；此时不打开查看器 |
| `--device` | 仿真设备，默认 `cuda:0` |

不指定 `--clip` 时，查看器从加载顺序中的第一个动作开始。若 checkpoint 使用多个动作，保留完整的 `--motion-file` 目录，用 `--clip` 选择其中一个。

### 5.2 在 Viser 中切换动作、选择帧与重置

同一次播放可以切换已加载的全部动作，不需要重新启动命令：

1. 点击 **Pause / 暂停**；运行过程中 Motion、Frame 和 Start Here 控件禁用。
2. 在 **Commands → Motion** 面板的动作下拉框中选择动作。
3. 将 **Frame** 设为希望初始化的帧 `k`，从头开始时设为 `0`。
4. 点击 **Start Here / 从这里开始**，应用动作和帧选择。
5. 点击 **Play / 继续**，恢复策略执行。

只修改下拉框或 Frame 不会立即改变机器人状态，需要点击 Start Here。下拉列表显示的是本次启动加载的动作，运行过程中新增 NPZ 文件不会自动加入列表。

Frame 使用动作内从零开始的初始化帧索引 `k`。机器人重置到参考第 `k` 帧，首次控制目标为 `k+1`，奖励和终止判断使用该次控制的目标帧。参考机器人仍显示当前目标，保留现有的 `k → k+1` 时序。若手动选择动作末帧，目标钳制在该末帧，不会越界到下一个动作。

查看器的 **Reset / 重置环境** 会从当前动作的首帧重新开始；播放中失败或动作结束后，同样从当前动作首帧重新播放，不会自动切到下一个动作。要从中间帧开始，应使用 Frame 和 Start Here，而不是普通 Reset。

### 5.3 导出策略执行 MP4

只导出一个动作：

```bash
python scripts/play_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --checkpoint-file logs/supermimic/g1_20261024_180000_123456/last.pt \
  --clip dance1_subject2 \
  --video-dir videos/supermimic/g1_20261024_180000_123456 \
  --device cuda:0
```

输出文件为 `videos/supermimic/g1_20261024_180000_123456/dance1_subject2.mp4`。不指定 `--clip` 时，依次为整个动作库导出视频：

```bash
python scripts/play_supermimic.py \
  --motion-file src/assets/motions/g1_npz \
  --checkpoint-file logs/supermimic/g1_20261024_180000_123456/last.pt \
  --video-dir videos/supermimic/g1_20261024_180000_123456 \
  --device cuda:0
```

每个动作从首帧初始化，录制一次，直到动作结束或策略失败，不循环拼接；提前失败的视频会相应缩短。视频使用 MP4/H.264，帧率为当前控制频率 50 Hz，文件名为 `<动作名称>.mp4`。已有同名视频会被重新写入，可按 checkpoint 或实验名使用不同输出目录。

添加 `--video-dir` 后进入离屏录制模式，即使同时设置 `--viewer` 也不会启动查看器。视频写入使用 `mediapy` 和 FFmpeg，运行环境需提供可用的离屏渲染与视频编码支持。

| 输出方式 | 内容 |
|---|---|
| `csv_to_npz_multi.py --save-video` | 数据转换时的参考动作视频，不加载策略 |
| `play_supermimic.py --video-dir` | checkpoint 策略实际控制机器人的执行视频 |
| `evaluate_supermimic.py --record` | 机器人状态轨迹 NPZ，用于数据分析，不是视频 |

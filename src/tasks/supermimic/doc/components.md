# SuperMimic 组件扩展

G1 的算法、网络选择和学习率仍在 `config/g1/rl_cfg.py` 配置，奖励与终止仍在环境配置中组合。训练主循环只负责采集、WM 更新、Actor 想象更新、Critic 更新及日志；它不按奖励名字分支。

本次拆分保留现有采样、优化器、归一化、损失归约、梯度裁剪、自举与参考时序：reset 到 k，目标 f=k+1；执行动作后使用 f 计算奖励和终止，再推进目标。带感知模式继续共享 CNN＋GRU，只有 WM 优化器更新共享参数。

## 1. 奖励有两种实现途径

### 可由状态、动作、参考和 FK 计算的奖励

在 `mdp/rewards.py` 新增纯 PyTorch 函数，用 `analytic_reward` 标记，然后在环境中配置：

```python
from ..core.components import analytic_reward

@analytic_reward
def joint_speed_l2(ctx):
  return ctx.state[5].square().sum(dim=-1)
```

```python
"joint_speed": RewardTermCfg(
  func=mdp.joint_speed_l2,
  weight=-0.01,
),
```

真实环境和想象轨迹调用同一函数。函数返回未乘权重、未乘 `step_dt` 的标量张量；聚合层负责加权。无需修改 Runner、经验池或训练算法。

`StepContext` 提供：

| 字段 | 内容 |
|---|---|
| `state` | `(p, R, v, omega, q, qd)`；R 为旋转矩阵，采用现有参考坐标表达 |
| `frame` | 本步当前目标帧 f |
| `action` / `previous_action` | 本步动作和上一动作，未应用执行器 action scale |
| `reference_root` | 当前目标根位置、旋转矩阵、线/角速度 |
| `bodies` | 当前状态经可微 FK 得到的身体位置、姿态和线/角速度 |
| `aligned_reference` | 现有身体跟踪对齐后的参考位置和姿态 |
| `reference_bodies` | NPZ 中原有的参考身体数据 |
| `terrain_tolerance` | 当前 episode 对应地形是否允许下肢容忍 |

FK 和参考数据只在当前计算内缓存，保留梯度，不跨 optimizer step 缓存。新函数要支持任意批次前缀，用 `...`、`dim=-1` 等表示最后的特征维，避免在可微路径中使用 NumPy、`.item()` 或 `.detach()`。

已有真实管理器函数也可用 `reward_component(tensor=..., prepare=...)` 注册对应的想象公式。`prepare(params, body_names)` 在构造时把身体名字等转换成索引；真实测量仍沿用原函数。现有上下肢奖励采用此方式，保留重构前的计算。

### 依赖接触传感器等信息的奖励

保留一个 `func(env, **params)` 计算真实原始标量，并为它声明预测监督：

```python
from ..core.components import Prediction, reward_component

# my_sensor_penalty 是实际的传感器计算函数，返回 [num_envs]。
my_sensor_penalty = reward_component(
  prediction=Prediction(loss_weight=1.0, nonnegative=True),
  sense_before_reward=True,  # 需要动作后重新 sense 的射线等传感器才设置
)(my_sensor_penalty)
```

所需传感器在具体任务工厂中显式创建，例如 `unitree_g1_terrain_env_cfg()` 中的 `cfg.scene.sensors`。奖励配置的 `sensor_name` 等参数必须与传感器名称一致。增删奖励时需要同步维护传感器配置，不再由奖励组件自动装配。

基础平地任务只使用 G1 公共传感器；Terrain 显式创建 `supermimic_depth`、`foot_collision_contacts`、`foot_support_contacts`、`foot_height_scanners`，训练与播放均保留。`collect_foot_rewards` 只控制监督标签采集，不增删传感器；奖励权重为 0 时仍可采集标签。

配置传感器并加入奖励函数后，预测声明继续负责：

1. 采集动作执行后、reset 前的原始标签。
2. 按配置顺序分配经验池标量列、WM 共享预测头的输出列。
3. 计算每项独立的 MSE 监督损失及 `wm_<奖励名>_loss` 日志。
4. 想象训练使用预测值 × 奖励权重 × `step_dt`。

默认共享头是“WM 最后隐藏特征 → 256 → 奖励数量”。非负惩罚使用 Softplus；有符号标量需声明 `nonnegative=False`。这里扩展的是标量奖励，向量监督需要另行定义对应接口。

独立监督权重可在 `rl_cfg.py` 设置：

```python
reward_loss_weights: dict[str, float] = field(default_factory=lambda: {
  "my_sensor_penalty": 0.5,
})
```

字典键使用环境配置中的奖励名字。没有覆盖时使用组件的默认值。原有 `feet_stumble_loss_weight` 和 `feet_at_plane_loss_weight` 继续有效；字典覆盖优先。权重为 0 的奖励仍可监督 WM；预测损失为 0 不代表奖励关闭，两者必须分开设置。

内部存储键 `foot_costs`、网络参数键 `foot_head` 和部分 `foot_reward_names` 属性沿用现有命名，但标签数量和名字已由组件配置决定，不再只支持两个脚部奖励。这些命名的保留不代表支持旧配置格式的 checkpoint。

非零奖励如果既无可微公式，也无预测声明，训练启动时会报出该奖励名，避免真实采集和想象训练悄悄使用不同目标。

## 2. 终止条件

物理失败使用“误差函数 + 阈值 + 软化宽度”：

```python
from ..core.components import analytic_failure

@analytic_failure(soft_width=0.05)
def excessive_joint_speed(ctx):
  return ctx.state[5].abs().amax(dim=-1)
```

```python
"joint_speed_failure": TerminationTermCfg(
  func=mdp.excessive_joint_speed,
  params={"threshold": 8.0},
),
```

真实环境使用 `error > threshold`；想象训练使用 `sigmoid((error-threshold)/soft_width)`。多个硬失败取 OR；软失败保持原来的 `1-product(1-d_i)`。删除一个配置项就同时移除它的硬失败和软失败，不需要在主循环中删第二份逻辑。

原有 `anchor_pos` 和 `anchor_ori` 已注册，高度/倾斜阈值及软化宽度未变。只要启用了需要高度修正的组件，才会准备参考地形高度；删除高度终止不会访问缺失的 `anchor_pos` 配置。

`episode_time_out` 和 `motion_clip_end` 是截断，仍使用 `time_out=True`，不计入上述软失败乘积。`episode_time_out` 带有时间上限标识，改配置名也能保持 bin 回报的原有有限长度口径。动作结束的可变展开、mask 和自举仍由训练算法维护，不应把失败伪装成截断。新增一种完全不同的任务结束语义，或依赖传感器的不可解析终止预测，需要专门设计想象对应路径；仅增加一个真实环境回调是不够的。

## 3. 网络替换

`networks/factory.py` 是唯一的构造入口，读取 `rl_cfg.py` 中的类路径和额外构造参数：

```python
actor_class: str = "src.tasks.supermimic.networks.my_actor:MyActor"
actor_kwargs: dict = field(default_factory=lambda: {"z_dim": 64})
critic_class: str = ".critic:Critic"
world_model_class: str = ".world_model:WorldModel"
# 以下两项只在 TerrainSuperMimicConfig 中声明：
perception_class: str = ".depth_encoder:DepthEncoder"
depth_decoder_class: str = ".depth_decoder:DepthDecoder"
```

以 `.` 开头的相对模块路径从 `supermimic.networks` 包解析。最直接的替换方式是继承现有网络、替换内部层，保留以下契约：

| 网络 | 训练层依赖的接口 |
|---|---|
| Actor | 现有构造签名；`forward(obs_history, goal, depth=None, depth_latent=None)`；`obs_norm`、`goal_norm`；输出 29 维动作 |
| Critic | 现有构造签名；`forward(features, depth=None, depth_latent=None)`；`norm`；每行输出一个价值 |
| WM | `inputs`、`targets`、`state_error`、`in_norm`、`out_norm`；`step`、`step_visual`、`step_with_foot_costs` 等当前动力学接口 |
| 感知编码器 | `encode_frames`、`aggregate`、`forward`，保持历史窗口及 h 的维度约定 |
| 深度解码器 | 输入 h，输出归一化的 18×32 深度图 |

工厂传入各网络已有的必需参数，例如 `hidden`、输入维度、`depth_cfg`；`*_kwargs` 用于额外参数，不要重复这些关键字。`wm_hidden`、`policy_hidden`、`critic_hidden` 继续控制默认网络宽度。

带感知模式中 Actor 和 Critic 必须使用工厂传入的同一个 `depth_encoder` 对象。共享参数从 Actor/Critic 优化器中排除，WM 与 Critic 在 BPTT 时只冻结参数，不切断输入梯度。替换类同样要遵守这一点。

这支持更换内部网络结构，不会自动兼容任意新的输入语义。例如增加另一种历史缓存或新的观测来源，还需扩展状态/特征契约；不应在网络类中读取仿真器来绕过想象训练。

训练、恢复、评估和播放均使用同一工厂，checkpoint 保存类路径、特征维度和奖励标签顺序。恢复要求这些结构一致；新网络结构应开始新实验。加载直接使用当前格式保存的配置，不再迁移旧配置字段。

## 4. 两种训练模式

```bash
# main 的无感知平地训练
python scripts/train_supermimic.py --motion-file /path/to/motions --task Unitree-G1-SuperMimic

# 当前分支的带感知地形训练，仍是默认值
python scripts/train_supermimic.py --motion-file /path/to/motions --task Unitree-G1-SuperMimic-Terrain
```

两个任务在 `config/g1/__init__.py` 注册，环境工厂位于 `env_cfgs.py`。基础平地任务保留原 main 的 8 项奖励和仿真容量，不创建深度传感器；Terrain 保留当前地形、感知和奖励配置，在任务工厂中添加 `DepthCameraCfg().sensor_cfg()`。两者共用 Runner 与训练算法。训练入口从环境的实际传感器确定网络配置的 `depth_enabled`；恢复、评估、播放按 checkpoint 的 `task_id` 创建环境，并要求其感知模式与保存的网络一致。统一使用 `--task` 指定完整任务名称。

训练配置使用数据类继承：`SuperMimicConfig` 提供普通无感知训练和通用算法参数，默认任务为 `Unitree-G1-SuperMimic`、`depth_enabled=False`；`TerrainSuperMimicConfig(SuperMimicConfig)` 继承通用参数，增加感知结构、感知学习率、表征/深度重建及脚部预测损失配置，默认选择 Terrain 任务并启用感知。`unitree_g1_runner_cfg()` 和 `unitree_g1_terrain_runner_cfg()` 分别返回基础类和子类实例。共享参数在基础类修改，Terrain 专属参数在子类修改。新训练省略 `--task` 时仍默认 Terrain，恢复时优先使用 checkpoint 保存的任务。

`tools.registry.runner_cfg_from_dict()` 根据 `task_id` 对应的注册工厂恢复具体配置类，再应用保存的字段；训练、恢复、评估和播放共用这一路径。普通配置不会携带感知字段，Terrain 配置会完整保存子类字段。新增配置子类只需由其任务工厂返回，无需在 Runner 或播放脚本中增加任务名称分支。

新增任务示例（放在相应机器人配置包中）：

```python
from ...tools.registry import register_task

register_task(
  "MyRobot-SuperMimic-Terrain",
  env_cfg_factory=".config.my_robot.env_cfgs:make_env_cfg",
  runner_cfg_factory=".config.my_robot.rl_cfg:make_runner_cfg",
)
```

工厂返回新的配置对象。环境工厂接受 `play`、`terrain_type`、`collect_foot_rewards`，所需传感器直接写入 `cfg.scene.sensors`；训练配置工厂返回 `SuperMimicConfig` 或其子类。`make_env()` 不再接受 `depth_enabled`，环境工厂也不再接收外部 `depth_camera`。确保注册包在启动前导入，然后通过 `--task` 选择。该注册表独立于 mjlab 的 PPO 注册表，原有 PPO 任务保持不变。

目录职责：`mdp` 只保留指令、观测、奖励、终止；`core` 负责状态和组件装配；`motion` 负责动作库及重置采样；`tools` 放任务注册、旋转与 FK；网络定义统一放在 `networks`。脚部公式已合入 `mdp/rewards.py`，高度/倾斜误差已合入 `mdp/terminations.py`。`rl/logging.py` 中的 `capture_step_metrics`、`IntervalMetrics`、`TrainingLogger` 分别负责快照、聚合和输出，调用时点保持不变。

用户可编辑的 G1 参数全部位于 `config/g1/rl_cfg.py`。已移除旧 checkpoint 配置迁移模块；废弃的 `wandb_name`、`actor_perception_lr`、`critic_perception_lr` 等配置字段不会再自动转换或忽略。

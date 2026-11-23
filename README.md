# KOPAL

KOPAL（Key-Object Point-cloud Action Link）是一个分层机器人操作模型。高层世界模型预测关键物体的固定对应点云轨迹，CEM 规划器在该轨迹空间中生成子目标，底层 MeanFlow 策略读取子目标轨迹并生成动作块。

世界模型与策略之间只传递形状为 `(B, H, O, N, 3)` 的关键物体点云轨迹，不在接口中插入潜变量或图像子目标。

代码仓库：[MetaverLei/KOPAL](https://github.com/MetaverLei/KOPAL.git)

## 模型结构

```text
RGB-D 点云与关键物体固定对应点
    -> 关键物体点流世界模型
    -> 物体级 SE(3) 动作条件
    -> CEM 规划
    -> 关键物体点云子目标轨迹 (B,H,O,N,3)
    -> 轨迹缓冲与时间推进
    -> MeanFlow 动作策略
    -> 1-NFE 动作块
```

### 世界模型

- 使用层次点云编码器提取逐点特征和全局特征。
- 在物体 PCA 规范坐标中预测固定对应点的未来位置。
- 输出由显式刚体动作和神经网络残差组成。
- JEPA 预测目标使用当前物体规范坐标，保留刚体相对运动。
- 默认在原始投影空间计算 JEPA 均方误差、方差和协方差，避免单位向量归一化导致方差目标不可达。
- 目标编码器和目标投影器使用指数滑动平均，不接收梯度。
- SIGReg 通过随机投影上的 Epps-Pulley 统计量约束 embedding 分布接近各向同性高斯分布，抑制低秩塌缩。
- 单步接口为 `step`，多步接口为 `rollout`，教师强制训练接口为 `forward`。

默认世界模型训练目标为：

```text
loss =
  1.0  * flow_loss
  + 1.0  * raw_JEPA_MSE
  + 5.0  * raw_variance_loss
  + 10.0 * raw_covariance_loss
  + 0.09 * SIGReg
```

训练日志分别记录 `wm_flow_loss`、`wm_jepa_loss`、`wm_jepa_weight`、
`wm_variance_loss`、`wm_covariance_loss`、`wm_normalized_variance_loss`、
`wm_normalized_covariance_loss`、`wm_latent_std_mean`、`wm_sigreg_loss`
和 `wm_total_loss`。JEPA 预测损失可通过 `jepa_warmup_epochs` 设置启用轮次，
归一化 JEPA 形式可通过 `jepa_normalize` 启用。

### 规划器

CEM 规划器在物体局部 SE(3) 空间采样动作，通过世界模型生成候选点云轨迹，并使用以下代价选择最优轨迹：

- 固定对应点终点误差；
- Chamfer 距离；
- 障碍碰撞；
- 轨迹平滑性；
- 物体之间交互。

规划器输出世界模型实际推演得到的点云轨迹，不输出独立于世界模型的抽象动作。

### 策略

MeanFlow 策略的输入包括：

- 场景点云；
- 机器人状态；
- 关键物体子目标点云轨迹；
- 当前关键物体点云；
- 固定对应 SE(3) token；
- 几何关系 token；
- 刚体残差较大时补充的局部点流 token。

动作 token 经过自注意力和条件交叉注意力后生成动作块。训练使用 MeanFlow 平均速度恒等式和端点回归损失，默认推理使用一次网络前向。

## 代码结构

| 文件 | 作用 |
|---|---|
| `kopal/config.py` | 编码器、世界模型、策略、规划器和感知配置 |
| `kopal/geometry.py` | SO(3)/SE(3)、刚体变换、PCA 规范坐标、固定点对应和点云采样 |
| `kopal/dataset.py` | 轨迹数据结构、滑动窗口、NPZ 读写和合成数据生成 |
| `kopal/modules.py` | 点云编码器、特征传播、时间编码、注意力和条件 Transformer 块 |
| `kopal/world_model.py` | 关键物体点流世界模型与 JEPA 训练目标 |
| `kopal/planner.py` | 物体级 SE(3) CEM 规划器 |
| `kopal/interface.py` | 世界模型到策略的子目标轨迹缓冲接口 |
| `kopal/policy.py` | 1-NFE MeanFlow 动作策略 |
| `kopal/system.py` | 世界模型、规划器、轨迹缓冲与策略的统一组合 |
| `kopal/perception.py` | RGB-D 反投影、Grounding DINO 检测、SAM2 分割和点云融合 |
| `kopal/metaworld_task.py` | MetaWorld 环境适配和脚本专家数据采集 |
| `kopal/metaworld_train.py` | MetaWorld 数据生成、训练与闭环评估命令 |
| `kopal/train.py` | 基于本地轨迹数据的分阶段训练命令 |

## 数据格式

离线训练文件采用 NPZ 或 PyTorch 字典格式，必须包含以下字段：

| 字段 | 形状 | 含义 |
|---|---:|---|
| `scene_points` | `(E,T,Ns,3)` | 场景点云 |
| `object_points` | `(E,T,O,No,3)` | 关键物体固定对应点云轨迹 |
| `robot_state` | `(E,T,Ds)` | 机器人状态 |
| `actions` | `(E,T,A)` | 连续动作 |
| `object_mask` | `(E,T,O)` | 可选物体有效性掩码 |
| `point_mask` | `(E,T,O,No)` | 可选逐点有效性掩码 |
| `episode_lengths` | `(E,)` | 可选有效轨迹长度 |

`object_points` 中的同一点编号在整个 episode 内必须保持同一物理对应点。

滑动窗口数据同时提供：

- `current_objects`：世界模型当前状态；
- `target_objects`：世界模型下一步监督；
- `subgoal_trajectory`：策略的未来轨迹条件；
- `actions`：策略动作块监督。

## 安装

克隆代码仓库：

```powershell
git clone https://github.com/MetaverLei/KOPAL.git
cd KOPAL
```

安装运行依赖：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

启用 Grounding DINO 与 SAM2 感知路径时安装可选依赖：

```powershell
python -m pip install -e ".[perception]"
```

真实 SAM2 推理还需要安装 `sam2` 包，并提供模型配置和 checkpoint。

## 训练本地轨迹数据

使用 NPZ 数据分阶段训练世界模型和策略：

```powershell
python -m kopal.train --stage world --data data/train.npz --steps 1000 --output outputs/world.pt
python -m kopal.train --stage policy --data data/train.npz --steps 1000 --output outputs/policy.pt
```

未提供 `--data` 时，训练入口使用确定性合成轨迹检查计算图和数据流。

## MetaWorld 使用

默认工作目录为项目根目录下的 `KOPAL_train`。数据保存在 `KOPAL_train/datasets`，模型输出保存在 `KOPAL_train/outputs`。

### 采集脚本专家数据

```powershell
python -m kopal.metaworld_train --mode generate --task-name reach-v3 --episodes 20 --seed 7
```

### 分阶段训练

```powershell
python -m kopal.metaworld_train --mode train --task-name reach-v3 --world-epochs 60 --policy-epochs 1000
```

训练入口支持以下关键参数：

- `--dataset`：指定 NPZ 数据路径；
- `--output-dir`：指定输出目录；
- `--batch-size`：训练批大小；
- `--world-lr`：世界模型学习率；
- `--policy-lr`：策略学习率；
- `--policy-weight-decay`：策略权重衰减；
- `--save-every`：checkpoint 保存间隔；
- `--resume`：从输出目录中的 `latest.pt` 恢复训练；
- `--device`：`cuda` 或 `cpu`。

### 闭环评估

```powershell
python -m kopal.metaworld_train --mode eval --task-name reach-v3 --checkpoint KOPAL_train/outputs/reach_v3/latest.pt --eval-episodes 20 --eval-seed 10000
```

评估参数包括：

- `--eval-max-steps`：单 episode 最大环境步数；
- `--replan-every`：世界模型与 CEM 重新规划间隔；
- `--approach-threshold`：接近阶段切换阈值；
- `--grasp-width-threshold`：抓取阶段夹爪宽度门限；
- `--use-policy-ema`：加载策略 EMA 参数。

## Python 调用

```python
import torch

from kopal import KOPALConfig
from kopal.system import KOPALSystem

config = KOPALConfig(device="cuda")
system = KOPALSystem(config).to(config.device)

system.update_subgoal(
    current_objects=torch.randn(1, 2, 64, 3, device=config.device),
    goal_objects=torch.randn(1, 2, 64, 3, device=config.device),
)

actions = system.act(
    scene_points=torch.randn(1, 2, 256, 3, device=config.device),
    robot_state=torch.randn(1, 2, 9, device=config.device),
)
```

`system.act` 返回动作块、完整动作预测、网络前向次数和当前子目标版本。

## 关键约束

- 关键物体点云轨迹必须保持固定点对应关系。
- 世界模型动作与规划器动作均使用物体级 SE(3) 增量。
- 策略条件中的子目标轨迹使用 `(B,H,O,N,3)` 形状。
- 策略动作维度和机器人状态维度由 `PolicyConfig` 决定。
- MetaWorld 配置使用 4 维动作、9 维机器人状态、16 步预测和 8 步执行。
- 感知模块可惰性加载 Grounding DINO 与 SAM2；预计算点云和掩码可直接进入训练流程。

"""KOPAL 模型配置定义。

本文件集中定义点云编码器、世界模型、阶段辅助头、MeanFlow 策略、CEM
规划器和感知模块的配置字段及默认值。所有组件通过同一份配置构建，避免
训练和推理阶段使用不一致的超参数。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class EncoderConfig:
    point_hidden_dims: tuple[int, ...] = (64, 128)
    token_dim: int = 128
    global_dim: int = 256
    abstract_points: tuple[int, ...] = (32, 12)
    knn: int = 12


@dataclass
class WorldModelConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    object_dim: int = 256
    transformer_layers: int = 3
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    dropout: float = 0.0
    residual_scale: float = 0.10
    use_jepa: bool = True
    jepa_dim: int = 192
    ema_momentum: float = 0.995
    jepa_use_ema_target: bool = True
    jepa_warmup_epochs: int = 0
    flow_loss_weight: float = 1.0
    jepa_loss_weight: float = 1.0
    jepa_normalize: bool = False
    rigid_loss_weight: float = 0.0
    latent_variance_weight: float = 5.0
    latent_covariance_weight: float = 10.0
    latent_regularizer_on_raw: bool = True
    regularize_encoder_feature: bool = False
    regularize_target_feature: bool = False
    normalized_variance_weight: float = 0.0
    normalized_covariance_weight: float = 0.0
    normalized_std_margin: float = 0.05
    sigreg_weight: float = 0.09
    sigreg_num_proj: int = 256
    sigreg_knots: int = 17
    sigreg_chunk_size: int = 128
    num_phase_types: int = 3
    phase_loss_weight: float = 1.0
    phase_boundary_loss_weight: float = 0.5
    phase_count_loss_weight: float = 0.5
    phase_context_dim: int = 9


@dataclass
class PolicyConfig:
    scene_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(abstract_points=(32, 12), token_dim=128, global_dim=256))
    object_encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(abstract_points=(16, 8), token_dim=96, global_dim=192))
    d_model: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    temporal_kernel_size: int = 5
    temporal_scale: float = 0.5
    dropout: float = 0.0
    action_dim: int = 7
    action_horizon: int = 8
    n_action_steps: int = 8
    n_obs_steps: int = 2
    robot_state_dim: int = 9
    num_local_flow_tokens: int = 8
    pose_token_scale: float = 5.0
    local_delta_scale: float = 10.0
    geometry_token_scale: float = 10.0
    rigid_residual_scale: float = 100.0
    rigid_residual_threshold: float = 0.005
    subgoal_augmentation_prob: float = 0.0
    subgoal_noise_std: float = 0.001
    meanflow_norm_p: float = 0.75
    meanflow_norm_eps: float = 1.0e-3
    meanflow_endpoint_weight: float = 5.0
    inference_noise: str = "zeros"
    meanflow_t_mean: float = -0.6
    meanflow_r_mean: float = -4.0
    meanflow_time_std: float = 1.6
    meanflow_different_interval_ratio: float = 0.75
    ema_decay: float = 0.999
    inference_steps: int = 1


@dataclass
class PlannerConfig:
    num_phase_types: int = 3
    phase_context_dim: int = 9
    horizon: int = 8
    num_keyframes: int = 1
    num_samples: int = 32
    num_elites: int = 8
    num_iterations: int = 3
    init_translation_std: float = 0.05
    init_rotation_std: float = 0.20
    min_translation_std: float = 0.005
    min_rotation_std: float = 0.02
    translation_cost_weight: float = 1.0
    collision_weight: float = 0.5
    collision_margin: float = 0.01
    fixed_correspondence_weight: float = 1.0
    chamfer_weight: float = 0.25
    smoothness_weight: float = 0.05
    object_interaction_weight: float = 0.20
    ema_alpha: float = 0.35
    use_warm_start: bool = True
    warm_start_decay: float = 0.8
    stage_cost_weight: float = 0.2


@dataclass
class KOPALConfig:
    object_count: int = 2
    object_points: int = 64
    scene_points: int = 256
    obstacle_points: int = 64
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    device: str = "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerceptionConfig:
    object_points: int = 64
    scene_points: int = 1024
    min_depth: float = 0.05
    max_depth: float = 5.0
    mask_threshold: float = 0.5
    depth_scale: float = 1.0
    ground_dino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_checkpoint: Optional[str] = None
    sam2_config: Optional[str] = None
    box_threshold: float = 0.35
    text_threshold: float = 0.25

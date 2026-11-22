"""KOPAL 的 1-NFE MeanFlow 动作块策略。

本文件将场景点云、机器人状态和世界模型生成的关键物体点云轨迹编码为条件
token，通过条件 Transformer 生成动作块，并使用 MeanFlow 平均速度恒等式
支持单次网络前向推理。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PolicyConfig
from .geometry import (
    estimate_rigid_transform,
    farthest_point_indices,
    gather_points,
    rt_to_action_6d,
    transform_points,
)
from .modules import (
    ConditionalTransformerBlock,
    PointNet2Encoder,
    PointNetGlobalEncoder,
    mlp,
    sinusoidal_time_embedding,
)


class MeanFlowPointPolicy(nn.Module):
    """以三维观测和关键物体子目标轨迹为条件的高频动作策略。

    条件 token 包含场景几何、机器人本体状态、每个未来位姿的固定对应
    SE(3) token、几何关系 token，以及刚体拟合残差较大时补充的局部点流
    token。每个条件 Transformer 层对动作 token 执行一次条件交叉注意力。
    MeanFlow 通过平均速度恒等式训练，默认使用一次网络前向生成动作块。
    """

    def __init__(self, config: Optional[PolicyConfig] = None) -> None:
        super().__init__()
        self.config = config or PolicyConfig()
        cfg = self.config
        self.scene_encoder = PointNet2Encoder(cfg.scene_encoder)
        self.object_encoder = PointNet2Encoder(cfg.object_encoder)
        self.flow_encoder = PointNet2Encoder(cfg.object_encoder)
        self.scene_proj = nn.Linear(cfg.scene_encoder.global_dim, cfg.d_model)
        self.object_proj = nn.Linear(cfg.object_encoder.global_dim, cfg.d_model)
        self.flow_global_proj = mlp(
            cfg.object_encoder.global_dim * 2,
            cfg.d_model,
            cfg.d_model,
            layers=2,
        )
        self.flow_local_proj = mlp(
            cfg.object_encoder.token_dim * 2 + 3,
            cfg.d_model,
            cfg.d_model,
            layers=2,
        )
        self.rigid_token_proj = mlp(7, cfg.d_model, cfg.d_model, layers=2)
        self.geometry_token_proj = mlp(15, cfg.d_model, cfg.d_model, layers=2)
        self.state_proj = mlp(cfg.robot_state_dim, cfg.d_model, cfg.d_model, layers=2)
        self.condition_type_proj = nn.Embedding(6, cfg.d_model)
        self.action_in = nn.Linear(cfg.action_dim, cfg.d_model)
        self.time_proj = mlp(64, cfg.d_model, cfg.d_model, layers=2)
        self.action_out = mlp(cfg.d_model, cfg.d_model, cfg.action_dim, layers=2)
        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(
                    cfg.d_model,
                    cfg.transformer_heads,
                    cfg.transformer_ff_dim,
                    cfg.dropout,
                )
                for _ in range(cfg.transformer_layers)
            ]
        )
        self.temporal_refine = nn.Sequential(
            nn.Conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.temporal_kernel_size,
                padding=cfg.temporal_kernel_size // 2,
            ),
            nn.GELU(),
            nn.Conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.temporal_kernel_size,
                padding=cfg.temporal_kernel_size // 2,
            ),
        )
        nn.init.zeros_(self.temporal_refine[-1].weight)
        nn.init.zeros_(self.temporal_refine[-1].bias)
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.register_buffer("action_mean", torch.zeros(cfg.action_dim))
        self.register_buffer("action_std", torch.ones(cfg.action_dim))
        self.register_buffer("action_min", -torch.ones(cfg.action_dim))
        self.register_buffer("action_max", torch.ones(cfg.action_dim))
        self.register_buffer(
            "action_binary_mask",
            torch.zeros(cfg.action_dim, dtype=torch.bool),
        )
        self.register_buffer("state_mean", torch.zeros(cfg.robot_state_dim))
        self.register_buffer("state_std", torch.ones(cfg.robot_state_dim))

    def set_action_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape != self.action_mean.shape or std.shape != self.action_std.shape:
            raise ValueError("Action statistics have the wrong shape")
        self.action_mean.copy_(mean)
        self.action_std.copy_(std.clamp_min(1.0e-6))

    def set_action_bounds(
        self,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
    ) -> None:
        if minimum.shape != self.action_min.shape or maximum.shape != self.action_max.shape:
            raise ValueError("Action bounds have the wrong shape")
        self.action_min.copy_(minimum)
        self.action_max.copy_(maximum)

    def set_action_binary_mask(self, mask: torch.Tensor) -> None:
        if mask.shape != self.action_binary_mask.shape:
            raise ValueError("Binary action mask has the wrong shape")
        self.action_binary_mask.copy_(mask.to(self.action_binary_mask.dtype))

    def set_state_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        if mean.shape != self.state_mean.shape or std.shape != self.state_std.shape:
            raise ValueError("State statistics have the wrong shape")
        self.state_mean.copy_(mean)
        self.state_std.copy_(std.clamp_min(1.0e-6))

    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            2.0
            * (actions - self.action_min)
            / (self.action_max - self.action_min).clamp_min(1.0e-6)
            - 1.0
        )

    def denormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return (
            (actions + 1.0)
            * 0.5
            * (self.action_max - self.action_min)
            + self.action_min
        )

    def _encode_conditions(
        self,
        scene_points: torch.Tensor,
        subgoal_trajectory: torch.Tensor,
        robot_state: torch.Tensor,
        scene_mask: Optional[torch.Tensor],
        subgoal_mask: Optional[torch.Tensor],
        subgoal_point_mask: Optional[torch.Tensor],
        current_objects: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scene_points.ndim != 4 or scene_points.shape[-1] != 3:
            raise ValueError(f"scene_points must be (B,To,Ns,3), got {tuple(scene_points.shape)}")
        if subgoal_trajectory.ndim != 5 or subgoal_trajectory.shape[-1] != 3:
            raise ValueError(
                "subgoal_trajectory must be (B,H,O,N,3), "
                f"got {tuple(subgoal_trajectory.shape)}"
            )
        if robot_state.ndim != 3:
            raise ValueError(f"robot_state must be (B,To,Ds), got {tuple(robot_state.shape)}")
        b, to, ns, _ = scene_points.shape
        _, h, o, no, _ = subgoal_trajectory.shape
        if current_objects is not None and current_objects.shape != (b, o, no, 3):
            raise ValueError(
                f"current_objects must be {(b, o, no, 3)}, "
                f"got {tuple(current_objects.shape)}"
            )
        if robot_state.shape[:2] != (b, to):
            raise ValueError("robot_state must share batch and observation steps with scene_points")

        scene_mask_flat = scene_mask.reshape(b * to, ns) if scene_mask is not None else None
        _, scene_feature = self.scene_encoder(scene_points.reshape(b * to, ns, 3), scene_mask_flat)
        scene_tokens = self.scene_proj(scene_feature).reshape(b, to, -1)
        scene_tokens = scene_tokens + self.condition_type_proj.weight[0]

        if subgoal_mask is not None and subgoal_mask.shape == (b, o):
            subgoal_mask = subgoal_mask[:, None].expand(-1, h, -1)
        if subgoal_point_mask is not None and subgoal_point_mask.shape == (b, o, no):
            subgoal_point_mask = subgoal_point_mask[:, None].expand(-1, h, -1, -1)
        object_mask_flat = (
            subgoal_mask.reshape(b * h * o)
            if subgoal_mask is not None
            else torch.ones(b * h * o, dtype=torch.bool, device=scene_points.device)
        )
        point_mask_flat = (
            subgoal_point_mask.reshape(b * h * o, no)
            if subgoal_point_mask is not None
            else torch.ones(b * h * o, no, dtype=torch.bool, device=scene_points.device)
        )
        anchor_objects = (
            current_objects
            if current_objects is not None
            else subgoal_trajectory[:, 0]
        )
        subgoal_flat = subgoal_trajectory.reshape(b * h * o, no, 3)
        anchor_flat = anchor_objects.reshape(b * o, no, 3)
        repeated_anchor = anchor_objects[:, None].expand(
            -1, h, -1, -1, -1
        )
        repeated_anchor_flat = repeated_anchor.reshape(b * h * o, no, 3)
        delta_flat = (
            subgoal_trajectory - repeated_anchor
        ).reshape(b * h * o, no, 3)

        target_per_point, target_feature = self.object_encoder(
            subgoal_flat,
            point_mask_flat,
        )
        delta_per_point, delta_feature = self.flow_encoder(
            delta_flat,
            point_mask_flat,
        )
        _, current_feature = self.object_encoder(anchor_flat)

        time_ids = torch.arange(
            h,
            dtype=torch.float32,
            device=scene_points.device,
        )
        time_tokens = sinusoidal_time_embedding(
            time_ids,
            self.config.d_model,
        )

        relative_rotation, relative_translation = estimate_rigid_transform(
            repeated_anchor_flat,
            subgoal_flat,
            mask=point_mask_flat,
        )
        rigid_twist = rt_to_action_6d(
            relative_rotation,
            relative_translation,
        )
        rigid_prediction = transform_points(
            repeated_anchor_flat,
            relative_rotation,
            relative_translation,
        )
        rigid_error = (rigid_prediction - subgoal_flat).square().sum(dim=-1)
        rigid_error = (
            rigid_error * point_mask_flat.to(rigid_error.dtype)
        ).sum(dim=-1) / point_mask_flat.to(rigid_error.dtype).sum(
            dim=-1
        ).clamp_min(1.0)
        rigid_error = torch.sqrt(rigid_error)
        rigid_tokens = self.rigid_token_proj(
            torch.cat(
                (
                    rigid_twist * self.config.pose_token_scale,
                    (
                        rigid_error * self.config.rigid_residual_scale
                    ).unsqueeze(-1),
                ),
                dim=-1,
            )
        ).reshape(b, h, o, -1)
        rigid_gate = torch.exp(
            -(
                rigid_error
                / self.config.rigid_residual_threshold
            ).square()
        ).reshape(b, h, o, 1)

        flow_globals = self.flow_global_proj(
            torch.cat((target_feature, delta_feature), dim=-1)
        ).reshape(b, h, o, -1)
        summary_tokens = (
            self.object_proj(target_feature).reshape(b, h, o, -1)
            + flow_globals
            + rigid_tokens
            + time_tokens.view(1, h, 1, -1)
            + self.condition_type_proj.weight[1]
        )
        summary_tokens = summary_tokens * object_mask_flat.reshape(
            b, h, o, 1
        ).to(summary_tokens.dtype)

        current_tokens = self.object_proj(current_feature).reshape(b, o, -1)
        current_tokens = current_tokens + self.condition_type_proj.weight[3]

        anchor_centers = anchor_objects.mean(dim=-2)
        subgoal_centers = subgoal_trajectory.mean(dim=-2)
        expanded_anchor = anchor_centers[:, None].expand(-1, h, -1, -1)
        hand_position = robot_state[:, -1, :3].view(b, 1, 1, 3).expand(
            -1,
            h,
            o,
            -1,
        )
        geometry_tokens = self.geometry_token_proj(
            torch.cat(
                (
                    expanded_anchor,
                    subgoal_centers,
                    subgoal_centers - expanded_anchor,
                    hand_position - expanded_anchor,
                    hand_position - subgoal_centers,
                ),
                dim=-1,
            )
            * self.config.geometry_token_scale
        )
        geometry_tokens = (
            geometry_tokens
            + time_tokens.view(1, h, 1, -1)
            + self.condition_type_proj.weight[5]
        )
        geometry_tokens = geometry_tokens * object_mask_flat.reshape(
            b,
            h,
            o,
            1,
        ).to(geometry_tokens.dtype)

        point_count = min(self.config.num_local_flow_tokens, no)
        index = farthest_point_indices(anchor_flat, point_count)
        shared_index = index.reshape(b, 1, o, point_count).expand(
            -1, h, -1, -1
        ).reshape(b * h * o, point_count)
        selected_target = gather_points(target_per_point, shared_index)
        selected_delta = gather_points(delta_per_point, shared_index)
        selected_delta_xyz = gather_points(delta_flat, shared_index)
        local_tokens = self.flow_local_proj(
            torch.cat(
                (
                    selected_target,
                    selected_delta,
                    selected_delta_xyz * self.config.local_delta_scale,
                ),
                dim=-1,
            )
        ).reshape(b, h, o, point_count, -1)
        local_tokens = (
            local_tokens
            + time_tokens.view(1, h, 1, 1, -1)
            + self.condition_type_proj.weight[4]
        )
        deformable_gate = 1.0 - rigid_gate
        local_tokens = (
            local_tokens
            * deformable_gate.unsqueeze(-2)
            * object_mask_flat.reshape(b, h, o, 1, 1).to(
                local_tokens.dtype
            )
        )

        normalized_state = (
            robot_state - self.state_mean
        ) / self.state_std
        state_tokens = (
            self.state_proj(normalized_state)
            + self.condition_type_proj.weight[2]
        )
        token_groups = [
            scene_tokens,
            current_tokens,
            summary_tokens.reshape(b, h * o, -1),
            geometry_tokens.reshape(b, h * o, -1),
            local_tokens.reshape(b, h * o * point_count, -1),
            state_tokens,
        ]
        valid_groups = [
            torch.ones(b, to, dtype=torch.bool, device=scene_points.device),
            object_mask_flat.reshape(b, h, o)[:, 0],
            object_mask_flat.reshape(b, h * o),
            object_mask_flat.reshape(b, h * o),
            (
                object_mask_flat.reshape(b, h, o, 1).expand(
                    -1, -1, -1, point_count
                ).reshape(b, h * o * point_count)
                & (
                    deformable_gate.reshape(b, h, o, 1).expand(
                        -1, -1, -1, point_count
                    ).reshape(b, h * o * point_count)
                    > 0.1
                )
            ),
            torch.ones(b, to, dtype=torch.bool, device=scene_points.device),
        ]
        condition_tokens = torch.cat(token_groups, dim=1)
        condition_valid = torch.cat(valid_groups, dim=1)
        return condition_tokens, ~condition_valid

    def predict_velocity(
        self,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
        interval: torch.Tensor,
        condition_tokens: torch.Tensor,
        condition_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, h, _ = noisy_action.shape
        if time.shape != (b,) or interval.shape != (b,):
            raise ValueError("time and interval must both have shape (B,)")
        x = self.action_in(noisy_action)
        time_input = torch.cat(
            (
                sinusoidal_time_embedding(time, 32),
                sinusoidal_time_embedding(interval, 32),
            ),
            dim=-1,
        )
        position = torch.arange(h, dtype=torch.float32, device=noisy_action.device)
        x = x + sinusoidal_time_embedding(position, self.config.d_model).unsqueeze(0)
        x = x + self.time_proj(time_input).unsqueeze(1)
        for block in self.blocks:
            x = block(x, condition_tokens, condition_mask=condition_padding_mask)
        temporal_residual = self.temporal_refine(
            x.transpose(1, 2)
        ).transpose(1, 2)
        x = x + self.config.temporal_scale * temporal_residual
        return self.action_out(self.final_norm(x))

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        current_objects = (
            batch["current_objects"][:, 0]
            if "current_objects" in batch
            else None
        )
        subgoal = batch["subgoal_trajectory"]
        if self.training and current_objects is not None:
            fractions = torch.linspace(
                1.0,
                0.0,
                subgoal.shape[1],
                device=subgoal.device,
                dtype=subgoal.dtype,
            ).flip(0).view(1, -1, 1, 1, 1)
            final_offset = subgoal[:, -1:] - current_objects[:, None]
            planner_like = current_objects[:, None] + fractions * final_offset
            planner_like = 0.75 * planner_like + 0.25 * subgoal
            use_planner_like = (
                torch.rand(
                    subgoal.shape[0],
                    1,
                    1,
                    1,
                    1,
                    device=subgoal.device,
                )
                < self.config.subgoal_augmentation_prob
            )
            subgoal = torch.where(use_planner_like, planner_like, subgoal)
            subgoal = subgoal + torch.randn_like(subgoal) * (
                self.config.subgoal_noise_std
            )
        conditions, condition_mask = self._encode_conditions(
            batch["scene_points"],
            subgoal,
            batch["robot_state"],
            batch.get("scene_mask"),
            batch.get("subgoal_mask"),
            batch.get("subgoal_point_mask"),
            current_objects=current_objects,
        )
        actions = self.normalize_actions(batch["actions"])
        if actions.ndim != 3:
            raise ValueError(f"actions must be (B,H,A), got {tuple(actions.shape)}")
        noise = torch.randn_like(actions)
        t = torch.sigmoid(
            torch.randn(actions.shape[0], device=actions.device)
            * self.config.meanflow_time_std
            + self.config.meanflow_t_mean
        )
        r = torch.sigmoid(
            torch.randn(actions.shape[0], device=actions.device)
            * self.config.meanflow_time_std
            + self.config.meanflow_r_mean
        )
        same_endpoint = (
            torch.rand_like(t)
            < 1.0 - self.config.meanflow_different_interval_ratio
        )
        r = torch.where(same_endpoint, t, r)
        r = torch.minimum(r, t)
        interval = t - r
        t_view = t.view(-1, 1, 1)
        interval_view = interval.view(-1, 1, 1)
        path_point = (1.0 - t_view) * actions + t_view * noise
        velocity = noise - actions

        def velocity_function(z: torch.Tensor, current_t: torch.Tensor, current_interval: torch.Tensor) -> torch.Tensor:
            return self.predict_velocity(
                z,
                current_t,
                current_interval,
                conditions,
                condition_mask,
            )

        with torch.autocast(device_type=actions.device.type, enabled=False):
            prediction, derivative = torch.func.jvp(
                velocity_function,
                (path_point, t, interval),
                (velocity, torch.ones_like(t), torch.zeros_like(interval)),
            )
            target_velocity = (velocity - interval_view * derivative).detach()
            per_sample = (prediction - target_velocity).square().flatten(1).sum(dim=-1)
            weight = (per_sample.detach() + self.config.meanflow_norm_eps).pow(self.config.meanflow_norm_p)
            meanflow_loss = (per_sample / weight).mean()
        endpoint_noise = torch.zeros_like(actions)
        endpoint_velocity = self.predict_velocity(
            endpoint_noise,
            torch.ones(actions.shape[0], device=actions.device),
            torch.ones(actions.shape[0], device=actions.device),
            conditions,
            condition_mask,
        )
        endpoint_prediction = endpoint_noise - endpoint_velocity
        endpoint_loss = F.mse_loss(endpoint_prediction, actions)
        loss = (
            meanflow_loss
            + self.config.meanflow_endpoint_weight * endpoint_loss
        )
        return loss, {
            "policy_meanflow_loss": meanflow_loss,
            "policy_endpoint_loss": endpoint_loss,
            "policy_total_loss": loss,
            "policy_velocity_error": F.mse_loss(prediction, target_velocity).detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        scene_points: torch.Tensor,
        subgoal_trajectory: torch.Tensor,
        robot_state: torch.Tensor,
        scene_mask: Optional[torch.Tensor] = None,
        subgoal_mask: Optional[torch.Tensor] = None,
        subgoal_point_mask: Optional[torch.Tensor] = None,
        current_objects: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        conditions, condition_mask = self._encode_conditions(
            scene_points,
            subgoal_trajectory,
            robot_state,
            scene_mask,
            subgoal_mask,
            subgoal_point_mask,
            current_objects=current_objects,
        )
        b = scene_points.shape[0]
        num_steps = self.config.inference_steps if num_steps is None else num_steps
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if self.config.inference_noise == "zeros":
            z = torch.zeros(
                b,
                self.config.action_horizon,
                self.config.action_dim,
                device=scene_points.device,
                dtype=scene_points.dtype,
            )
        elif self.config.inference_noise == "gaussian":
            z = torch.randn(
                b,
                self.config.action_horizon,
                self.config.action_dim,
                device=scene_points.device,
                dtype=scene_points.dtype,
                generator=generator,
            )
        else:
            raise ValueError(
                f"Unknown inference_noise={self.config.inference_noise!r}"
            )
        t = torch.ones(b, device=scene_points.device, dtype=scene_points.dtype)
        for _ in range(num_steps):
            r = (t - 1.0 / num_steps).clamp_min(0.0)
            interval = t - r
            u = self.predict_velocity(z, t, interval, conditions, condition_mask)
            z = z - interval.view(-1, 1, 1) * u
            t = r
        actions = self.denormalize_actions(z)
        if self.action_binary_mask.any():
            midpoint = 0.5 * (self.action_min + self.action_max)
            high = self.action_max.expand_as(actions)
            low = self.action_min.expand_as(actions)
            binary = torch.where(
                actions > midpoint,
                high,
                low,
            )
            actions = torch.where(
                self.action_binary_mask.expand_as(actions),
                binary,
                actions,
            )
        return {
            "action": actions[:, : self.config.n_action_steps],
            "action_pred": actions,
            "num_function_evaluations": torch.tensor(num_steps, device=scene_points.device),
        }

    def predict_action(self, batch: Dict[str, torch.Tensor], num_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        return self.sample(
            batch["scene_points"],
            batch["subgoal_trajectory"],
            batch["robot_state"],
            batch.get("scene_mask"),
            batch.get("subgoal_mask"),
            batch.get("subgoal_point_mask"),
            current_objects=(
                batch["current_objects"][:, 0]
                if "current_objects" in batch
                else None
            ),
            num_steps=num_steps,
        )


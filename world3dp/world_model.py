"""KOPAL 的关键物体三维点流世界模型。

本文件在物体局部规范坐标中预测固定对应点的未来位置，结合显式刚体位姿和
学习残差描述物体动力学，并支持 JEPA 潜空间预测、目标编码器指数滑动平均
以及潜变量方差和协方差正则化。
"""

from __future__ import annotations

import copy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WorldModelConfig
from .geometry import (
    CanonicalPointCloud,
    action_6d_to_rt,
    canonicalize_objects,
    estimate_object_action,
    pairwise_distance_loss,
    rt_to_action_6d,
    world_action_to_local,
)
from .modules import (
    ConditionalTransformerBlock,
    PointNet2Encoder,
    masked_smooth_l1,
    mlp,
    sinusoidal_time_embedding,
)


class ObjectPointFlowWorldModel(nn.Module):
    """根据物体级 SE(3) 命令预测固定对应关键物体的未来点云。

    前向转移形式为 ``O_hat[t+1] = WM(O[t], Delta[t])``。网络在确定性的
    物体 PCA 坐标系中工作，将 SE(3) 刚体命令与学习残差相加得到局部未来
    位置，再变换回世界坐标。该结构显式保留刚体运动语义，同时允许表达非
    刚性残差。
    """

    def __init__(self, config: Optional[WorldModelConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldModelConfig()
        self.encoder = PointNet2Encoder(self.config.encoder)
        in_dim = self.config.encoder.token_dim
        object_dim = self.config.object_dim
        self.point_proj = nn.Linear(in_dim, object_dim)
        self.action_encoder = mlp(6 + 32, object_dim, object_dim, layers=2)
        self.object_norm = nn.LayerNorm(object_dim)
        self.object_context = nn.ModuleList(
            [
                ConditionalTransformerBlock(
                    object_dim,
                    self.config.transformer_heads,
                    self.config.transformer_ff_dim,
                    self.config.dropout,
                )
                for _ in range(self.config.transformer_layers)
            ]
        )
        self.decoder = mlp(object_dim * 2 + 6 + 4, object_dim, 3, layers=3)

        self.latent_projector = mlp(
            self.config.encoder.global_dim,
            self.config.jepa_dim,
            self.config.jepa_dim,
            layers=2,
        )
        if self.config.use_jepa:
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_latent_projector = copy.deepcopy(self.latent_projector)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            for parameter in self.target_latent_projector.parameters():
                parameter.requires_grad_(False)
        else:
            self.target_encoder = None
            self.target_latent_projector = None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
            self.target_latent_projector.eval()
        return self

    @staticmethod
    def _expand_leading(points: torch.Tensor) -> tuple[torch.Tensor, bool]:
        single_step = points.ndim == 4
        if single_step:
            points = points.unsqueeze(1)
        if points.ndim != 5 or points.shape[-1] != 3:
            raise ValueError(f"Expected objects (B,T,O,N,3) or (B,O,N,3), got {tuple(points.shape)}")
        return points, single_step

    def _encode_objects(
        self,
        local_points: torch.Tensor,
        point_mask: Optional[torch.Tensor],
        use_target: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.target_encoder if use_target else self.encoder
        if encoder is None:
            raise RuntimeError("Target encoder was disabled by config")
        per_point, global_feature = encoder(local_points, point_mask)
        return per_point, global_feature

    def step(
        self,
        current_objects: torch.Tensor,
        actions: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """使用物体级 SE(3) 旋量执行一步未来点云预测。

        参数：
            current_objects：形状为 ``(B,O,N,3)`` 的世界坐标固定对应点。
            actions：形状为 ``(B,O,6)`` 的世界坐标 SE(3) 动作。
            object_mask：形状为 ``(B,O)`` 的物体有效性掩码。
            point_mask：形状为 ``(B,O,N)`` 的点有效性掩码。
        """
        if current_objects.ndim != 4:
            raise ValueError(f"step expects (B,O,N,3), got {tuple(current_objects.shape)}")
        b, o, n = current_objects.shape[:3]
        if actions.shape != (b, o, 6):
            raise ValueError(f"actions must be {(b, o, 6)}, got {tuple(actions.shape)}")
        if object_mask is None:
            object_mask = torch.ones(b, o, dtype=torch.bool, device=current_objects.device)
        if point_mask is None:
            point_mask = torch.ones(b, o, n, dtype=torch.bool, device=current_objects.device)

        canonical = canonicalize_objects(current_objects, object_mask & point_mask.any(dim=-1))
        local_points = canonical.points
        world_rotation, world_translation = action_6d_to_rt(actions)
        local_rotation, local_translation = world_action_to_local(
            world_rotation,
            world_translation,
            canonical.frame,
        )
        local_action = rt_to_action_6d(local_rotation, local_translation)

        per_point, _ = self._encode_objects(local_points, point_mask)
        point_tokens = self.point_proj(per_point)
        action_context = self.action_encoder(
            torch.cat((local_action, sinusoidal_time_embedding(torch.ones_like(local_action[..., 0]), 32)), dim=-1)
        )
        object_tokens = point_tokens.max(dim=-2).values + action_context
        object_tokens = self.object_norm(object_tokens)
        invalid_object_mask = ~object_mask
        for block in self.object_context:
            object_tokens = block(
                object_tokens,
                action_context,
                condition_mask=invalid_object_mask,
                self_mask=invalid_object_mask,
            )
        context = object_tokens.unsqueeze(-2).expand(-1, -1, n, -1)
        action_points = local_action.unsqueeze(-2).expand(-1, -1, n, -1)
        decoder_input = torch.cat(
            (
                point_tokens,
                context,
                action_points,
                local_points,
                local_points.square().sum(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
        residual = self.config.residual_scale * torch.tanh(self.decoder(decoder_input))
        predicted_local = local_points @ local_rotation.transpose(-1, -2) + local_translation.unsqueeze(-2) + residual
        predicted_world = canonical.to_world(predicted_local)
        predicted_world = predicted_world * point_mask.to(predicted_world.dtype).unsqueeze(-1)

        output = {
            "next_objects": predicted_world,
            "predicted_local": predicted_local,
            "residual": residual,
            "object_tokens": object_tokens,
            "local_actions": local_action,
        }
        if return_latent:
            valid_points = point_mask.any(dim=-1).to(predicted_local.dtype).unsqueeze(-1)
            _, future_embedding = self._encode_objects(predicted_local, point_mask)
            output["predicted_embedding"] = self.latent_projector(future_embedding) * valid_points
        return output

    def rollout(
        self,
        current_objects: torch.Tensor,
        actions: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """根据 ``(B,T,O,6)`` 世界坐标动作自回归推演未来点云轨迹。"""
        if actions.ndim != 4 or actions.shape[-1] != 6:
            raise ValueError(f"actions must be (B,T,O,6), got {tuple(actions.shape)}")
        b, t, o, _ = actions.shape
        if current_objects.shape[:2] != (b, o):
            raise ValueError("current_objects and actions disagree on batch/object dimensions")
        current = current_objects
        trajectory = []
        residuals = []
        for k in range(t):
            output = self.step(current, actions[:, k], object_mask=object_mask, point_mask=point_mask)
            current = output["next_objects"]
            trajectory.append(current)
            residuals.append(output["residual"])
        return {
            "objects": torch.stack(trajectory, dim=1),
            "residuals": torch.stack(residuals, dim=1),
        }

    def forward(
        self,
        current_objects: torch.Tensor,
        target_objects: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """执行教师强制约束下的多步前向预测。

        ``current_objects`` 和 ``target_objects`` 的形状均为
        ``(B,T,O,N,3)``。未提供动作时，从固定对应点轨迹中估计每步刚体
        动作。动作使用世界坐标旋量，并在每一步内部转换到物体当前局部坐标。
        """
        current, _ = self._expand_leading(current_objects)
        if target_objects is None:
            raise ValueError("target_objects must be provided for teacher-forced forward")
        target, _ = self._expand_leading(target_objects)
        if target.shape != current.shape:
            raise ValueError("current_objects and target_objects must have equal shapes")
        b, t, o, n, _ = current.shape
        if object_mask is None:
            object_mask = torch.ones(b, t, o, dtype=torch.bool, device=current.device)
        if point_mask is None:
            point_mask = torch.ones(b, t, o, n, dtype=torch.bool, device=current.device)
        if actions is None:
            inferred_rotation, inferred_translation = estimate_object_action(
                current.reshape(b * t * o, n, 3),
                target.reshape(b * t * o, n, 3),
                mask=point_mask.reshape(b * t * o, n),
            )
            actions = rt_to_action_6d(
                inferred_rotation,
                inferred_translation,
            ).reshape(b, t, o, 6)
        if actions.shape != (b, t, o, 6):
            raise ValueError(f"actions must be {(b, t, o, 6)}, got {tuple(actions.shape)}")

        predictions = []
        residuals = []
        predicted_embeddings = []
        for k in range(t):
            output = self.step(
                current[:, k],
                actions[:, k],
                object_mask=object_mask[:, k],
                point_mask=point_mask[:, k],
                return_latent=return_latent,
            )
            predictions.append(output["next_objects"])
            residuals.append(output["residual"])
            if return_latent:
                predicted_embeddings.append(output["predicted_embedding"])

        result = {
            "next_objects": torch.stack(predictions, dim=1),
            "residuals": torch.stack(residuals, dim=1),
            "actions": actions,
        }
        if return_latent:
            result["predicted_embedding"] = torch.stack(predicted_embeddings, dim=1)
        return result

    def compute_loss(
        self,
        current_objects: torch.Tensor,
        target_objects: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        rigid_object_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算几何回归、无未来泄漏的 JEPA 预测损失和潜变量正则项。"""
        current, _ = self._expand_leading(current_objects)
        target, _ = self._expand_leading(target_objects)
        b, t, o, n, _ = current.shape
        if object_mask is None:
            object_mask = torch.ones(b, t, o, dtype=torch.bool, device=current.device)
        if point_mask is None:
            point_mask = torch.ones(b, t, o, n, dtype=torch.bool, device=current.device)
        use_latent = self.config.use_jepa
        output = self.forward(
            current,
            target,
            actions=actions,
            object_mask=object_mask,
            point_mask=point_mask,
            return_latent=use_latent,
        )
        pred = output["next_objects"]
        valid_points = point_mask & object_mask.unsqueeze(-1)
        flow_loss = masked_smooth_l1(pred - current, target - current, valid_points, beta=0.02)
        total = self.config.flow_loss_weight * flow_loss
        losses = {
            "wm_flow_loss": flow_loss,
        }

        if self.config.rigid_loss_weight > 0.0:
            if rigid_object_mask is None:
                rigid_object_mask = object_mask
            rigid = pairwise_distance_loss(
                pred.reshape(b * t * o, n, 3),
                target.reshape(b * t * o, n, 3),
            )
            total = total + self.config.rigid_loss_weight * rigid
            losses["wm_rigid_loss"] = rigid

        if use_latent:
            canonical_target = canonicalize_objects(target, object_mask & point_mask.any(dim=-1))
            with torch.no_grad():
                _, target_feature = self._encode_objects(canonical_target.points, point_mask, use_target=True)
                target_embedding = self.target_latent_projector(target_feature)
                target_embedding = F.normalize(target_embedding, dim=-1)
            predicted_embedding = F.normalize(output["predicted_embedding"], dim=-1)
            valid_objects = object_mask.to(predicted_embedding.dtype).unsqueeze(-1)
            token_loss = ((predicted_embedding - target_embedding).square().sum(dim=-1, keepdim=True) * valid_objects).sum()
            token_loss = token_loss / valid_objects.sum().clamp_min(1.0)
            total = total + self.config.jepa_loss_weight * token_loss
            losses["wm_jepa_loss"] = token_loss

            flat_embedding = predicted_embedding.reshape(-1, predicted_embedding.shape[-1])
            flat_valid = object_mask.reshape(-1)
            flat_embedding = flat_embedding[flat_valid]
            if flat_embedding.shape[0] > 1:
                std = torch.sqrt(flat_embedding.var(dim=0, unbiased=False) + 1.0e-4)
                variance_loss = F.relu(1.0 - std).mean()
                centered = flat_embedding - flat_embedding.mean(dim=0, keepdim=True)
                covariance = centered.transpose(-1, -2) @ centered / max(flat_embedding.shape[0] - 1, 1)
                off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
                covariance_loss = off_diagonal.square().sum() / flat_embedding.shape[-1]
                if self.config.latent_variance_weight > 0.0:
                    total = total + self.config.latent_variance_weight * variance_loss
                if self.config.latent_covariance_weight > 0.0:
                    total = total + self.config.latent_covariance_weight * covariance_loss
                losses["wm_variance_loss"] = variance_loss
                losses["wm_covariance_loss"] = covariance_loss
        losses["wm_total_loss"] = total
        return total, losses

    @torch.no_grad()
    def update_target_encoder(self, momentum: Optional[float] = None) -> None:
        if self.target_encoder is None:
            return
        momentum = self.config.ema_momentum if momentum is None else momentum
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(),
            self.encoder.buffers(),
        ):
            target.copy_(online)
        for target, online in zip(
            self.target_latent_projector.parameters(),
            self.latent_projector.parameters(),
        ):
            target.mul_(momentum).add_(online, alpha=1.0 - momentum)


PointFlowWorldModel = ObjectPointFlowWorldModel


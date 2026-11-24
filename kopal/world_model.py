"""KOPAL 的关键物体三维点流世界模型。

本文件在物体局部规范坐标中预测固定对应点的未来位置，结合显式刚体位姿和
学习残差描述物体动力学，并通过阶段辅助头预测当前阶段、阶段边界和终止
阶段编号。世界模型同时支持 JEPA 潜空间预测、目标编码器指数滑动平均以及
潜变量方差和协方差正则化。
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
    SketchIsotropicGaussianRegularizer,
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
        # 单独调用损失函数时默认启用非 warmup 目标，训练入口会通过 set_epoch 更新。
        self.current_epoch = 1
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
        self.phase_head = mlp(
            object_dim,
            object_dim,
            self.config.num_phase_types,
            layers=2,
        )
        self.phase_boundary_head = mlp(object_dim, object_dim, 1, layers=2)
        self.phase_count_head = mlp(
            object_dim,
            object_dim,
            self.config.num_phase_types,
            layers=2,
        )
        self.phase_context_proj = mlp(
            self.config.phase_context_dim,
            object_dim,
            object_dim,
            layers=2,
        )

        self.latent_projector = mlp(
            self.config.encoder.global_dim,
            self.config.jepa_dim,
            self.config.jepa_dim,
            layers=2,
        )
        self.sigreg = (
            SketchIsotropicGaussianRegularizer(
                knots=self.config.sigreg_knots,
                num_proj=self.config.sigreg_num_proj,
            )
            if self.config.sigreg_weight > 0.0
            else None
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

    def set_epoch(self, epoch: int) -> None:
        """设置训练轮次，用于控制损失项的 warmup 计划。"""
        self.current_epoch = int(epoch)

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
        phase_context: Optional[torch.Tensor] = None,
        time_index: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """使用物体级 SE(3) 旋量执行一步未来点云预测。

        参数：
            current_objects：形状为 ``(B,O,N,3)`` 的世界坐标固定对应点。
            actions：形状为 ``(B,O,6)`` 的世界坐标 SE(3) 动作。
            object_mask：形状为 ``(B,O)`` 的物体有效性掩码。
            point_mask：形状为 ``(B,O,N)`` 的点有效性掩码。
            phase_context：形状为 ``(B,Ds)`` 的当前机器人或阶段上下文。
            time_index：形状为 ``(B,O)`` 的归一化 rollout 时间索引。
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
        if phase_context is None:
            phase_context = torch.zeros(
                b,
                self.config.phase_context_dim,
                dtype=current_objects.dtype,
                device=current_objects.device,
            )
        elif phase_context.shape != (b, self.config.phase_context_dim):
            raise ValueError(
                f"phase_context must be {(b, self.config.phase_context_dim)}, "
                f"got {tuple(phase_context.shape)}"
            )

        valid_object_mask = object_mask & point_mask.any(dim=-1)
        canonical = canonicalize_objects(current_objects, valid_object_mask)
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
        if time_index is None:
            time_index = torch.zeros_like(local_action[..., 0])
        action_context = self.action_encoder(
            torch.cat(
                (
                    local_action,
                    sinusoidal_time_embedding(time_index, 32),
                ),
                dim=-1,
            )
        )
        object_tokens = (
            point_tokens.max(dim=-2).values
            + action_context
            + self.phase_context_proj(phase_context).unsqueeze(1)
        )
        object_tokens = self.object_norm(object_tokens)
        invalid_object_mask = ~valid_object_mask
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
            "phase_logits": self.phase_head(object_tokens),
            "phase_boundary_logits": self.phase_boundary_head(object_tokens).squeeze(-1),
            "phase_count_logits": self.phase_count_head(object_tokens[:, 0]),
        }
        if return_latent:
            valid_points = point_mask.any(dim=-1).to(predicted_local.dtype).unsqueeze(-1)
            _, future_embedding = self._encode_objects(predicted_local, point_mask)
            output["predicted_encoder_feature"] = future_embedding * valid_points
            output["predicted_embedding"] = self.latent_projector(future_embedding) * valid_points
        return output

    def rollout(
        self,
        current_objects: torch.Tensor,
        actions: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        phase_context: Optional[torch.Tensor] = None,
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
        phase_logits = []
        phase_boundary_logits = []
        for k in range(t):
            output = self.step(
                current,
                actions[:, k],
                object_mask=object_mask,
                point_mask=point_mask,
                phase_context=(
                    phase_context[:, k] if phase_context is not None else None
                ),
                time_index=torch.full(
                    (b, o),
                    k / max(t - 1, 1),
                    dtype=current.dtype,
                    device=current.device,
                ),
            )
            current = output["next_objects"]
            trajectory.append(current)
            residuals.append(output["residual"])
            phase_logits.append(output["phase_logits"])
            phase_boundary_logits.append(output["phase_boundary_logits"])
        return {
            "objects": torch.stack(trajectory, dim=1),
            "residuals": torch.stack(residuals, dim=1),
            "phase_logits": torch.stack(phase_logits, dim=1),
            "phase_boundary_logits": torch.stack(phase_boundary_logits, dim=1),
            "phase_count_logits": output["phase_count_logits"],
        }

    def forward(
        self,
        current_objects: torch.Tensor,
        target_objects: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        phase_context: Optional[torch.Tensor] = None,
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
        if phase_context is None:
            phase_context = torch.zeros(
                b,
                t,
                self.config.phase_context_dim,
                dtype=current.dtype,
                device=current.device,
            )
        elif phase_context.shape != (b, t, self.config.phase_context_dim):
            raise ValueError(
                f"phase_context must be {(b, t, self.config.phase_context_dim)}, "
                f"got {tuple(phase_context.shape)}"
            )
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
        predicted_encoder_features = []
        phase_logits = []
        phase_boundary_logits = []
        for k in range(t):
            output = self.step(
                current[:, k],
                actions[:, k],
                object_mask=object_mask[:, k],
                point_mask=point_mask[:, k],
                phase_context=phase_context[:, k],
                time_index=torch.full(
                    (b, o),
                    k / max(t - 1, 1),
                    dtype=current.dtype,
                    device=current.device,
                ),
                return_latent=return_latent,
            )
            predictions.append(output["next_objects"])
            residuals.append(output["residual"])
            phase_logits.append(output["phase_logits"])
            phase_boundary_logits.append(output["phase_boundary_logits"])
            if return_latent:
                predicted_embeddings.append(output["predicted_embedding"])
                predicted_encoder_features.append(output["predicted_encoder_feature"])

        result = {
            "next_objects": torch.stack(predictions, dim=1),
            "residuals": torch.stack(residuals, dim=1),
            "actions": actions,
            "phase_logits": torch.stack(phase_logits, dim=1),
            "phase_boundary_logits": torch.stack(phase_boundary_logits, dim=1),
            "phase_count_logits": output["phase_count_logits"],
        }
        if return_latent:
            result["predicted_embedding"] = torch.stack(predicted_embeddings, dim=1)
            result["predicted_encoder_feature"] = torch.stack(
                predicted_encoder_features, dim=1
            )
        return result

    def compute_loss(
        self,
        current_objects: torch.Tensor,
        target_objects: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        rigid_object_mask: Optional[torch.Tensor] = None,
        phase_id: Optional[torch.Tensor] = None,
        phase_boundary: Optional[torch.Tensor] = None,
        phase_context: Optional[torch.Tensor] = None,
        phase_count: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算几何回归、阶段辅助损失、JEPA 预测损失和潜变量正则项。"""
        current, _ = self._expand_leading(current_objects)
        target, _ = self._expand_leading(target_objects)
        b, t, o, n, _ = current.shape
        if object_mask is None:
            object_mask = torch.ones(b, t, o, dtype=torch.bool, device=current.device)
        if point_mask is None:
            point_mask = torch.ones(b, t, o, n, dtype=torch.bool, device=current.device)
        if phase_context is not None and phase_context.shape == (
            b,
            self.config.phase_context_dim,
        ):
            phase_context = phase_context[:, None].expand(-1, t, -1)
        use_latent = self.config.use_jepa
        output = self.forward(
            current,
            target,
            actions=actions,
            object_mask=object_mask,
            point_mask=point_mask,
            phase_context=phase_context,
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

        valid_object_mask = object_mask & point_mask.any(dim=-1)
        if phase_id is not None:
            phase_target = phase_id.long().clamp(
                0,
                self.config.num_phase_types - 1,
            )
            phase_loss = F.cross_entropy(
                output["phase_logits"][valid_object_mask],
                phase_target[valid_object_mask],
            )
            total = total + self.config.phase_loss_weight * phase_loss
            losses["wm_phase_loss"] = phase_loss
        if phase_boundary is not None:
            boundary_logits = output["phase_boundary_logits"][valid_object_mask]
            boundary_target = phase_boundary.to(boundary_logits.dtype)[
                valid_object_mask
            ]
            boundary_loss = F.binary_cross_entropy_with_logits(
                boundary_logits,
                boundary_target,
            )
            total = total + self.config.phase_boundary_loss_weight * boundary_loss
            losses["wm_phase_boundary_loss"] = boundary_loss
        if phase_count is not None:
            count_target = (phase_count.long() - 1).clamp(
                0,
                self.config.num_phase_types - 1,
            )
            count_loss = F.cross_entropy(
                output["phase_count_logits"],
                count_target,
            )
            total = total + self.config.phase_count_loss_weight * count_loss
            losses["wm_phase_count_loss"] = count_loss

        if use_latent:
            # 预测点位于当前物体的规范坐标中，未来目标也必须使用当前物体的
            # 规范坐标。若对目标独立执行 PCA 规范化，会消除刚体相对运动。
            valid_object_mask = object_mask & point_mask.any(dim=-1)
            current_canonical = canonicalize_objects(current, valid_object_mask)
            target_in_current_frame = current_canonical.to_local(target)
            if self.config.jepa_use_ema_target:
                with torch.no_grad():
                    _, target_feature = self._encode_objects(
                        target_in_current_frame,
                        point_mask,
                        use_target=True,
                    )
                    target_raw_embedding = self.target_latent_projector(target_feature)
                online_target_feature = None
                if self.config.regularize_target_feature:
                    _, online_target_feature = self._encode_objects(
                        target_in_current_frame,
                        point_mask,
                        use_target=False,
                    )
            else:
                _, online_target_feature = self._encode_objects(
                    target_in_current_frame,
                    point_mask,
                    use_target=False,
                )
                target_raw_embedding = self.latent_projector(online_target_feature)

            target_embedding = F.normalize(target_raw_embedding, dim=-1)
            predicted_raw_embedding = output["predicted_embedding"]
            predicted_embedding = F.normalize(predicted_raw_embedding, dim=-1)
            valid_objects = valid_object_mask.to(
                predicted_embedding.dtype
            ).unsqueeze(-1)
            if self.config.jepa_normalize:
                token_error = (
                    predicted_embedding - target_embedding
                ).square().sum(dim=-1, keepdim=True)
            else:
                token_error = (
                    predicted_raw_embedding - target_raw_embedding
                ).square().mean(dim=-1, keepdim=True)
            token_loss = (token_error * valid_objects).sum()
            token_loss = token_loss / valid_objects.sum().clamp_min(1.0)
            jepa_active = self.current_epoch > self.config.jepa_warmup_epochs
            jepa_weight = self.config.jepa_loss_weight if jepa_active else 0.0
            total = total + jepa_weight * token_loss
            losses["wm_jepa_loss"] = token_loss
            losses["wm_jepa_weight"] = torch.as_tensor(
                jepa_weight,
                device=token_loss.device,
                dtype=token_loss.dtype,
            )

            regularized_embedding = (
                predicted_raw_embedding
                if self.config.latent_regularizer_on_raw
                else predicted_embedding
            )
            regularized_embeddings = [regularized_embedding]
            if self.config.regularize_encoder_feature:
                regularized_embeddings.append(output["predicted_encoder_feature"])
            if self.config.regularize_target_feature:
                if online_target_feature is None:
                    _, online_target_feature = self._encode_objects(
                        target_in_current_frame,
                        point_mask,
                        use_target=False,
                    )
                regularized_embeddings.append(online_target_feature)

            flat_valid = valid_object_mask.reshape(-1)
            flat_embeddings = []
            for embedding in regularized_embeddings:
                flat = embedding.reshape(-1, embedding.shape[-1])
                flat_embeddings.append(flat[flat_valid])

            if flat_embeddings[0].shape[0] > 1:
                std_values = [
                    torch.sqrt(flat.var(dim=0, unbiased=False) + 1.0e-4)
                    for flat in flat_embeddings
                ]
                variance_losses = []
                covariance_losses = []
                normalized_variance_losses = []
                normalized_covariance_losses = []
                for flat, std in zip(flat_embeddings, std_values):
                    variance_losses.append(F.relu(1.0 - std).mean())
                    centered = flat - flat.mean(dim=0, keepdim=True)
                    covariance = centered.transpose(-1, -2) @ centered / max(
                        flat.shape[0] - 1,
                        1,
                    )
                    off_diagonal = covariance - torch.diag_embed(
                        covariance.diagonal()
                    )
                    covariance_losses.append(
                        off_diagonal.square().sum()
                        / max(flat.shape[-1] - 1, 1)
                        / flat.shape[-1]
                    )

                    normalized = F.normalize(flat, dim=-1)
                    normalized_std = torch.sqrt(
                        normalized.var(dim=0, unbiased=False) + 1.0e-4
                    )
                    normalized_variance_losses.append(
                        F.relu(
                            self.config.normalized_std_margin - normalized_std
                        ).mean()
                    )
                    normalized_centered = normalized - normalized.mean(
                        dim=0,
                        keepdim=True,
                    )
                    normalized_covariance = (
                        normalized_centered.transpose(-1, -2)
                        @ normalized_centered
                        / max(normalized.shape[0] - 1, 1)
                    )
                    normalized_off_diagonal = (
                        normalized_covariance
                        - torch.diag_embed(normalized_covariance.diagonal())
                    )
                    normalized_covariance_losses.append(
                        normalized_off_diagonal.square().sum()
                        / max(normalized.shape[-1] - 1, 1)
                        / normalized.shape[-1]
                    )

                variance_loss = torch.stack(variance_losses).mean()
                covariance_loss = torch.stack(covariance_losses).mean()
                normalized_variance_loss = torch.stack(
                    normalized_variance_losses
                ).mean()
                normalized_covariance_loss = torch.stack(
                    normalized_covariance_losses
                ).mean()
                if self.config.latent_variance_weight > 0.0:
                    total = (
                        total
                        + self.config.latent_variance_weight * variance_loss
                    )
                if self.config.latent_covariance_weight > 0.0:
                    total = (
                        total
                        + self.config.latent_covariance_weight * covariance_loss
                    )
                if self.config.normalized_variance_weight > 0.0:
                    total = (
                        total
                        + self.config.normalized_variance_weight
                        * normalized_variance_loss
                    )
                if self.config.normalized_covariance_weight > 0.0:
                    total = (
                        total
                        + self.config.normalized_covariance_weight
                        * normalized_covariance_loss
                    )

                losses["wm_variance_loss"] = variance_loss
                losses["wm_covariance_loss"] = covariance_loss
                losses["wm_normalized_variance_loss"] = normalized_variance_loss
                losses["wm_normalized_covariance_loss"] = normalized_covariance_loss
                losses["wm_latent_std_mean"] = torch.stack(
                    [std.mean() for std in std_values]
                ).mean()

                if self.sigreg is not None:
                    chunk_size = max(self.config.sigreg_chunk_size, 2)
                    sigreg_values = []
                    for flat in flat_embeddings:
                        sigreg_values.extend(
                            self.sigreg(chunk.unsqueeze(0))
                            for chunk in flat.split(chunk_size, dim=0)
                            if chunk.shape[0] > 1
                        )
                    sigreg_loss = torch.stack(sigreg_values).mean()
                    total = total + self.config.sigreg_weight * sigreg_loss
                    losses["wm_sigreg_loss"] = sigreg_loss
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


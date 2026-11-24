"""KOPAL 关键物体点云轨迹的 SE(3) CEM 规划器。

本文件在物体局部 SE(3) 空间采样候选位移，通过世界模型推演未来点云轨迹，
再结合终点几何距离、碰撞、平滑性、物体间交互和阶段一致性代价选择最优
子目标轨迹，并将选中的轨迹编码为阶段化关键物体点云轨迹。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .config import PlannerConfig
from .geometry import (
    action_6d_to_rt,
    canonicalize_objects,
    interpolate_keyframe_trajectory,
    local_action_to_world,
    rt_to_action_6d,
)
from .phases import build_staged_trajectory, decode_monotonic_phases
from .world_model import ObjectPointFlowWorldModel


@dataclass
class PlanningResult:
    subgoal_trajectory: torch.Tensor
    world_actions: torch.Tensor
    local_action: torch.Tensor
    total_costs: torch.Tensor
    terminal_costs: torch.Tensor
    collision_costs: torch.Tensor
    smoothness_costs: torch.Tensor
    interaction_costs: torch.Tensor
    stage_costs: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    phase_ids: Optional[torch.Tensor] = None
    phase_boundaries: Optional[torch.Tensor] = None
    phase_goal_points: Optional[torch.Tensor] = None
    phase_count: Optional[torch.Tensor] = None

    def named(self) -> Dict[str, torch.Tensor]:
        return {
            "subgoal_trajectory": self.subgoal_trajectory,
            "world_actions": self.world_actions,
            "local_action": self.local_action,
            "total_costs": self.total_costs,
            "terminal_costs": self.terminal_costs,
            "collision_costs": self.collision_costs,
            "smoothness_costs": self.smoothness_costs,
            "interaction_costs": self.interaction_costs,
            "stage_costs": self.stage_costs,
            "mean": self.mean,
            "std": self.std,
            "phase_ids": self.phase_ids,
            "phase_boundaries": self.phase_boundaries,
            "phase_goal_points": self.phase_goal_points,
            "phase_count": self.phase_count,
        }


class ObjectTrajectoryCEMPlanner:
    """使用交叉熵方法优化每个物体 SE(3) 轨迹的多步模型预测控制器。"""

    def __init__(
        self,
        world_model: ObjectPointFlowWorldModel,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.world_model = world_model
        self.config = config or PlannerConfig()
        self._previous_mean: Optional[torch.Tensor] = None

    def reset(self) -> None:
        self._previous_mean = None

    @staticmethod
    def _symmetric_chamfer(
        prediction: torch.Tensor,
        target: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算两组点云的对称 Chamfer 距离。"""
        distance = torch.cdist(prediction, target)
        if point_mask is None:
            prediction_term = distance.min(dim=-1).values.mean(dim=-1)
        else:
            nearest_target = distance.min(dim=-1).values
            prediction_term = (
                nearest_target * point_mask.to(nearest_target.dtype)
            ).sum(dim=-1) / point_mask.to(nearest_target.dtype).sum(dim=-1).clamp_min(1.0)
        if target_mask is None:
            target_term = distance.min(dim=-2).values.mean(dim=-1)
        else:
            nearest_prediction = distance.min(dim=-2).values
            target_term = (
                nearest_prediction * target_mask.to(nearest_prediction.dtype)
            ).sum(dim=-1) / target_mask.to(nearest_prediction.dtype).sum(dim=-1).clamp_min(1.0)
        return prediction_term + target_term

    def _initial_distribution(self, batch: int, objects: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.zeros(
            batch,
            objects,
            self.config.num_keyframes,
            6,
            device=device,
            dtype=dtype,
        )
        if self.config.use_warm_start and self._previous_mean is not None:
            previous = self._previous_mean.to(device=device, dtype=dtype)
            if previous.shape == mean.shape:
                mean = previous * self.config.warm_start_decay
        std = torch.empty_like(mean)
        std[..., :3] = self.config.init_rotation_std
        std[..., 3:] = self.config.init_translation_std
        return mean, std

    def _score(
        self,
        trajectory: torch.Tensor,
        goal_objects: torch.Tensor,
        obstacle_points: Optional[torch.Tensor],
        object_mask: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
        phase_logits: Optional[torch.Tensor] = None,
        phase_boundary_logits: Optional[torch.Tensor] = None,
        phase_count_logits: Optional[torch.Tensor] = None,
        max_phases: Optional[int] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch, samples, horizon, objects, points, _ = trajectory.shape
        goal = goal_objects[:, None].expand(-1, samples, -1, -1, -1)
        terminal = self._symmetric_chamfer(
            trajectory[:, :, -1],
            goal,
            point_mask=point_mask,
            target_mask=point_mask,
        )
        valid_objects = object_mask.to(terminal.dtype)
        terminal = (terminal * valid_objects).sum(dim=-1) / valid_objects.sum(dim=-1).clamp_min(1.0)
        correspondence = (trajectory[:, :, -1] - goal).square().sum(dim=-1)
        if point_mask is not None:
            correspondence = (
                correspondence * point_mask.to(correspondence.dtype)
            ).sum(dim=-1) / point_mask.to(correspondence.dtype).sum(
                dim=-1
            ).clamp_min(1.0)
        else:
            correspondence = correspondence.mean(dim=-1)
        correspondence = (
            correspondence * valid_objects
        ).sum(dim=-1) / valid_objects.sum(dim=-1).clamp_min(1.0)
        terminal = (
            self.config.fixed_correspondence_weight * correspondence
            + self.config.chamfer_weight * terminal
        )

        centroids = trajectory.mean(dim=-2)
        if horizon >= 3:
            acceleration = centroids[:, :, 2:] - 2.0 * centroids[:, :, 1:-1] + centroids[:, :, :-2]
            smoothness = acceleration.square().sum(dim=-1).mean(dim=(-1, -2))
        else:
            smoothness = trajectory.new_zeros(batch, samples)

        coll = trajectory.new_zeros(batch, samples)
        if obstacle_points is not None and obstacle_points.numel() > 0:
            obstacle = obstacle_points[:, None, None, None].expand(
                -1,
                samples,
                horizon,
                objects,
                -1,
                -1,
            )
            distance = torch.cdist(trajectory, obstacle)
            near = F.relu(self.config.collision_margin - distance.min(dim=-1).values)
            if point_mask is not None:
                near = near * point_mask.unsqueeze(2).to(near.dtype)
            coll = (near * object_mask[:, None, None, :, None].to(near.dtype)).sum(dim=(-1, -2, -3))
            valid_point_count = (
                point_mask.sum(dim=-1).to(coll.dtype)
                if point_mask is not None
                else near.new_full((batch, objects), points)
            )
            valid_total = (
                valid_point_count * object_mask.to(valid_point_count.dtype)
            ).sum(dim=-1, keepdim=True)
            coll = coll / (horizon * valid_total.clamp_min(1.0).squeeze(-1))

        interaction = trajectory.new_zeros(batch, samples)
        if objects > 1:
            pair_distance = torch.cdist(centroids, centroids)
            eye = torch.eye(objects, dtype=torch.bool, device=centroids.device).view(1, 1, objects, objects)
            pair_distance = pair_distance.masked_fill(eye, float("inf"))
            interaction_pairs = F.relu(self.config.collision_margin - pair_distance)
            pair_mask = object_mask[:, None, None, :, None] & object_mask[:, None, None, None, :]
            interaction = (
                interaction_pairs * pair_mask.to(interaction_pairs.dtype)
            ).sum(dim=(-1, -2, -3)) / pair_mask.sum(dim=(-1, -2, -3)).clamp_min(1)

        stage_cost = trajectory.new_zeros(batch, samples)
        if phase_logits is not None and max_phases != 1:
            discrete_phase = (
                decode_monotonic_phases(
                    phase_logits,
                    phase_count_logits,
                    max_phases=max_phases,
                )
                if phase_count_logits is not None
                else phase_logits.argmax(dim=-1)
            )
            valid = object_mask[:, None, None, :].to(discrete_phase.dtype)
            valid_points = valid.expand(-1, samples, horizon, -1)
            valid_total = valid_points.sum(dim=-1).clamp_min(1.0)
            if horizon > 1:
                phase_change = (
                    discrete_phase[:, :, 1:] != discrete_phase[:, :, :-1]
                ).to(trajectory.dtype)
                valid_change = valid_points[:, :, 1:]
                nonmonotonic = F.relu(
                    discrete_phase[:, :, :-1] - discrete_phase[:, :, 1:]
                )
                nonmonotonic = (
                    nonmonotonic * valid_change
                ).sum(dim=(-1, -2)) / valid_change.sum(
                    dim=(-1, -2)
                ).clamp_min(1.0)
                if phase_boundary_logits is not None:
                    boundary_error = F.binary_cross_entropy_with_logits(
                        phase_boundary_logits[:, :, :-1],
                        phase_change,
                        reduction="none",
                    )
                    boundary_error = (
                        boundary_error * valid_change
                    ).sum(dim=(-1, -2)) / valid_change.sum(
                        dim=(-1, -2)
                    ).clamp_min(1.0)
                else:
                    boundary_error = stage_cost
            else:
                nonmonotonic = stage_cost
                boundary_error = stage_cost

            centres = trajectory.mean(dim=-2)
            goal_centres = goal_objects.mean(dim=-2)[:, None]
            distance = torch.linalg.vector_norm(
                centres - goal_centres.unsqueeze(2),
                dim=-1,
            )
            distance = (
                distance * valid_points
            ).sum(dim=-1) / valid_total
            if horizon > 1:
                progress_violation = F.relu(
                    distance[:, :, 1:] - distance[:, :, :-1] + 1.0e-4
                ).mean(dim=-1)
            else:
                progress_violation = stage_cost

            if phase_count_logits is not None:
                expected_terminal = phase_count_logits.argmax(dim=-1)
                phase_max = discrete_phase.max(dim=2).values
                actual_terminal = phase_max.to(trajectory.dtype).mean(dim=-1)
                motion = torch.linalg.vector_norm(
                    centres[:, :, -1] - centres[:, :, 0],
                    dim=-1,
                )
                active_motion = (motion > 0.01).to(trajectory.dtype)
                coverage_violation = F.relu(
                    expected_terminal.to(trajectory.dtype).unsqueeze(-1)
                    - actual_terminal.unsqueeze(-1)
                ) * active_motion
                coverage_violation = coverage_violation.mean(dim=-1)
            else:
                coverage_violation = stage_cost

            stage_cost = (
                nonmonotonic
                + boundary_error
                + progress_violation
                + coverage_violation
            )

        total = (
            self.config.translation_cost_weight * terminal
            + self.config.collision_weight * coll
            + self.config.smoothness_weight * smoothness
            + self.config.object_interaction_weight * interaction
            + self.config.stage_cost_weight * stage_cost
        )
        return total, terminal, coll, smoothness, interaction, stage_cost

    @torch.no_grad()
    def plan(
        self,
        current_objects: torch.Tensor,
        goal_objects: torch.Tensor,
        obstacle_points: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        phase_context: Optional[torch.Tensor] = None,
        max_phases: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> PlanningResult:
        """返回最优世界模型点云轨迹及对应的物体级执行动作。"""
        if current_objects.ndim != 4 or current_objects.shape[-1] != 3:
            raise ValueError(f"current_objects must be (B,O,N,3), got {tuple(current_objects.shape)}")
        if goal_objects.shape != current_objects.shape:
            raise ValueError("goal_objects must preserve object count, point count, and correspondence")
        b, o, n, _ = current_objects.shape
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
        device, dtype = current_objects.device, current_objects.dtype
        frames = canonicalize_objects(current_objects, object_mask & point_mask.any(dim=-1)).frame
        mean, std = self._initial_distribution(b, o, device, dtype)
        best_cost: Optional[torch.Tensor] = None
        best_trajectory = None
        best_world_actions = None
        best_local_action = None
        best_phase_logits = None
        best_phase_boundary_logits = None
        best_phase_count_logits = None
        best_terms = None
        num_samples = max(self.config.num_elites + 1, self.config.num_samples)
        num_elites = min(self.config.num_elites, num_samples)

        for _ in range(self.config.num_iterations):
            noisy = mean[:, None] + std[:, None] * torch.randn(
                b,
                num_samples,
                o,
                self.config.num_keyframes,
                6,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            noisy[:, 0] = mean
            local_rotation, local_translation = action_6d_to_rt(noisy)
            world_rotation, world_translation = local_action_to_world(
                local_rotation,
                local_translation,
                frames[:, None, :, None],
            )
            delta_rotation, delta_translation = interpolate_keyframe_trajectory(
                world_rotation,
                world_translation,
                self.config.horizon,
            )
            world_actions = rt_to_action_6d(delta_rotation, delta_translation)

            flat_batch = b * num_samples
            flattened_actions = world_actions.reshape(flat_batch, self.config.horizon, o, 6)
            repeated_objects = current_objects[:, None].expand(-1, num_samples, -1, -1, -1).reshape(flat_batch, o, n, 3)
            repeated_object_mask = object_mask[:, None].expand(-1, num_samples, -1).reshape(flat_batch, o)
            repeated_point_mask = point_mask[:, None].expand(-1, num_samples, -1, -1).reshape(flat_batch, o, n)
            repeated_phase_context = phase_context[:, None, None].expand(
                -1,
                num_samples,
                self.config.horizon,
                -1,
            ).reshape(flat_batch, self.config.horizon, self.config.phase_context_dim)
            world_output = self.world_model.rollout(
                repeated_objects,
                flattened_actions,
                object_mask=repeated_object_mask,
                point_mask=repeated_point_mask,
                phase_context=repeated_phase_context,
            )
            rollout = world_output["objects"].reshape(
                b,
                num_samples,
                self.config.horizon,
                o,
                n,
                3,
            )
            rollout_phase_logits = world_output["phase_logits"].reshape(
                b,
                num_samples,
                self.config.horizon,
                o,
                self.config.num_phase_types,
            )
            rollout_phase_boundary_logits = world_output[
                "phase_boundary_logits"
            ].reshape(
                b,
                num_samples,
                self.config.horizon,
                o,
            )
            rollout_phase_count_logits = world_output[
                "phase_count_logits"
            ].reshape(
                b,
                num_samples,
                self.config.num_phase_types,
            )
            costs = self._score(
                rollout,
                goal_objects,
                obstacle_points,
                object_mask,
                point_mask=point_mask[:, None].expand(-1, num_samples, -1, -1),
                phase_logits=rollout_phase_logits,
                phase_boundary_logits=rollout_phase_boundary_logits,
                phase_count_logits=rollout_phase_count_logits,
                max_phases=max_phases,
            )
            total_costs = costs[0]
            elite_index = total_costs.topk(num_elites, dim=1, largest=False).indices
            elite_actions = torch.gather(
                noisy,
                1,
                elite_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    -1,
                    -1,
                    o,
                    self.config.num_keyframes,
                    6,
                ),
            )
            new_mean = elite_actions.mean(dim=1)
            new_std = elite_actions.std(dim=1, unbiased=False)
            floor = torch.empty_like(new_std)
            floor[..., :3] = self.config.min_rotation_std
            floor[..., 3:] = self.config.min_translation_std
            std = new_std.clamp_min(floor)
            mean = new_mean

            candidate_index = total_costs.argmin(dim=1)
            candidate_cost = total_costs.gather(1, candidate_index[:, None]).squeeze(1)

            def gather_candidate(values: torch.Tensor) -> torch.Tensor:
                batch_index = torch.arange(values.shape[0], device=values.device)
                return values[batch_index, candidate_index]

            if best_cost is None:
                best_cost = candidate_cost.detach()
                best_trajectory = gather_candidate(rollout).detach()
                best_world_actions = gather_candidate(world_actions).detach()
                best_local_action = gather_candidate(noisy).detach()
                best_phase_logits = gather_candidate(
                    rollout_phase_logits
                ).detach()
                best_phase_boundary_logits = gather_candidate(
                    rollout_phase_boundary_logits
                ).detach()
                best_phase_count_logits = gather_candidate(
                    rollout_phase_count_logits
                ).detach()
                best_terms = tuple(gather_candidate(term).detach() for term in costs[1:])
            else:
                improved = candidate_cost < best_cost
                if improved.any():
                    trajectory_candidate = gather_candidate(rollout)
                    world_candidate = gather_candidate(world_actions)
                    local_candidate = gather_candidate(noisy)
                    term_candidates = tuple(gather_candidate(term) for term in costs[1:])
                    choose = improved.view(-1, *([1] * (trajectory_candidate.ndim - 1)))
                    best_trajectory = torch.where(choose, trajectory_candidate.detach(), best_trajectory)
                    choose = improved.view(-1, *([1] * (world_candidate.ndim - 1)))
                    best_world_actions = torch.where(choose, world_candidate.detach(), best_world_actions)
                    choose = improved.view(-1, *([1] * (local_candidate.ndim - 1)))
                    best_local_action = torch.where(choose, local_candidate.detach(), best_local_action)
                    phase_candidate = gather_candidate(rollout_phase_logits)
                    choose = improved.view(
                        -1,
                        *([1] * (phase_candidate.ndim - 1)),
                    )
                    best_phase_logits = torch.where(
                        choose,
                        phase_candidate.detach(),
                        best_phase_logits,
                    )
                    boundary_candidate = gather_candidate(
                        rollout_phase_boundary_logits
                    )
                    choose = improved.view(
                        -1,
                        *([1] * (boundary_candidate.ndim - 1)),
                    )
                    best_phase_boundary_logits = torch.where(
                        choose,
                        boundary_candidate.detach(),
                        best_phase_boundary_logits,
                    )
                    count_candidate = gather_candidate(
                        rollout_phase_count_logits
                    )
                    choose = improved.view(
                        -1,
                        *([1] * (count_candidate.ndim - 1)),
                    )
                    best_phase_count_logits = torch.where(
                        choose,
                        count_candidate.detach(),
                        best_phase_count_logits,
                    )
                    updated_terms = []
                    for old, new in zip(best_terms, term_candidates):
                        choose = improved.view(-1, *([1] * (new.ndim - 1)))
                        updated_terms.append(torch.where(choose, new.detach(), old))
                    best_terms = tuple(updated_terms)
                    best_cost = torch.where(improved, candidate_cost.detach(), best_cost)

        if best_trajectory is None or best_cost is None or best_terms is None:
            raise RuntimeError("CEM planner did not produce a valid candidate")
        phase_ids = decode_monotonic_phases(
            best_phase_logits[:, None],
            best_phase_count_logits[:, None],
            max_phases=max_phases,
        )[:, 0]
        (
            staged_trajectory,
            phase_goal_points,
            phase_boundaries,
            phase_count,
        ) = build_staged_trajectory(best_trajectory, phase_ids)
        self._previous_mean = mean.detach()
        terminal, coll, smooth, interaction, stage = best_terms
        return PlanningResult(
            subgoal_trajectory=staged_trajectory,
            world_actions=best_world_actions,
            local_action=best_local_action,
            total_costs=best_cost,
            terminal_costs=terminal,
            collision_costs=coll,
            smoothness_costs=smooth,
            interaction_costs=interaction,
            stage_costs=stage,
            mean=mean.detach(),
            std=std.detach(),
            phase_ids=phase_ids,
            phase_boundaries=phase_boundaries,
            phase_goal_points=phase_goal_points,
            phase_count=phase_count,
        )


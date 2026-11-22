"""KOPAL 关键物体点云轨迹的 SE(3) CEM 规划器。

本文件在物体局部 SE(3) 空间采样候选位移，通过世界模型推演未来点云轨迹，
再结合终点几何距离、碰撞、平滑性和物体间交互代价选择最优子目标轨迹。
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
    mean: torch.Tensor
    std: torch.Tensor

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
            "mean": self.mean,
            "std": self.std,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        total = (
            self.config.translation_cost_weight * terminal
            + self.config.collision_weight * coll
            + self.config.smoothness_weight * smoothness
            + self.config.object_interaction_weight * interaction
        )
        return total, terminal, coll, smoothness, interaction

    @torch.no_grad()
    def plan(
        self,
        current_objects: torch.Tensor,
        goal_objects: torch.Tensor,
        obstacle_points: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
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
        device, dtype = current_objects.device, current_objects.dtype
        frames = canonicalize_objects(current_objects, object_mask & point_mask.any(dim=-1)).frame
        mean, std = self._initial_distribution(b, o, device, dtype)
        best_cost: Optional[torch.Tensor] = None
        best_trajectory = None
        best_world_actions = None
        best_local_action = None
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
            rollout = self.world_model.rollout(
                repeated_objects,
                flattened_actions,
                object_mask=repeated_object_mask,
                point_mask=repeated_point_mask,
            )["objects"].reshape(b, num_samples, self.config.horizon, o, n, 3)
            costs = self._score(
                rollout,
                goal_objects,
                obstacle_points,
                object_mask,
                point_mask=point_mask[:, None].expand(-1, num_samples, -1, -1),
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
                    updated_terms = []
                    for old, new in zip(best_terms, term_candidates):
                        choose = improved.view(-1, *([1] * (new.ndim - 1)))
                        updated_terms.append(torch.where(choose, new.detach(), old))
                    best_terms = tuple(updated_terms)
                    best_cost = torch.where(improved, candidate_cost.detach(), best_cost)

        if best_trajectory is None or best_cost is None or best_terms is None:
            raise RuntimeError("CEM planner did not produce a valid candidate")
        self._previous_mean = mean.detach()
        terminal, coll, smooth, interaction = best_terms
        return PlanningResult(
            subgoal_trajectory=best_trajectory,
            world_actions=best_world_actions,
            local_action=best_local_action,
            total_costs=best_cost,
            terminal_costs=terminal,
            collision_costs=coll,
            smoothness_costs=smooth,
            interaction_costs=interaction,
            mean=mean.detach(),
            std=std.detach(),
        )


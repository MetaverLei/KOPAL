"""KOPAL 刚体几何与固定对应点云工具。

所有点张量采用 ``(..., N, 3)`` 形状和行向量约定。刚体变换
``p' = p @ R.transpose(-1, -2) + t`` 由旋转矩阵 ``R`` 和平移 ``t``
组成。世界模型、规划器和策略统一使用该约定。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F


_EPS = 1.0e-8


def skew(w: torch.Tensor) -> torch.Tensor:
    """将 ``(..., 3)`` 旋转向量转换为反对称矩阵 ``(..., 3, 3)``。"""
    zero = torch.zeros_like(w[..., 0])
    return torch.stack(
        (
            torch.stack((zero, -w[..., 2], w[..., 1]), dim=-1),
            torch.stack((w[..., 2], zero, -w[..., 0]), dim=-1),
            torch.stack((-w[..., 1], w[..., 0], zero), dim=-1),
        ),
        dim=-2,
    )


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """将 ``(..., 3)`` 旋转向量通过指数映射转换为 SO(3) 旋转矩阵。"""
    theta2 = (omega * omega).sum(dim=-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2.clamp_min(_EPS))
    omega_hat = skew(omega)
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand_as(omega_hat)
    a = torch.where(theta2 < 1.0e-10, 1.0 - theta2 / 6.0, torch.sin(theta) / theta)
    b = torch.where(
        theta2 < 1.0e-10,
        0.5 - theta2 / 24.0,
        (1.0 - torch.cos(theta)) / theta2,
    )
    return eye + a * omega_hat + b * (omega_hat @ omega_hat)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """将 SO(3) 旋转矩阵通过对数映射转换为 ``(..., 3)`` 旋转向量。"""
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace - 1.0) * 0.5).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    theta = torch.acos(cos_theta)
    theta_safe = theta.clamp_min(1.0e-6)
    vee = torch.stack(
        (
            rotation[..., 2, 1] - rotation[..., 1, 2],
            rotation[..., 0, 2] - rotation[..., 2, 0],
            rotation[..., 1, 0] - rotation[..., 0, 1],
        ),
        dim=-1,
    )
    regular = vee * (theta / (2.0 * torch.sin(theta_safe))).unsqueeze(-1)

    # 接近零旋转时，反对称部分即可给出稳定极限。
    near_zero = 0.5 * vee

    # 接近 pi 时从对称部分恢复旋转轴。精确 pi 属于零测度情形，
    # 该分支用于保持采样和往返转换的数值稳定性。
    near_pi = theta > (torch.pi - 1.0e-3)
    if near_pi.any():
        diagonal = rotation.diagonal(dim1=-2, dim2=-1)
        axis = torch.sqrt((diagonal + 1.0).clamp_min(_EPS) * 0.5)
        # 选择绝对值最大的分量，并由非对角项确定符号，
        # 避免除以接近零的分量。
        largest = axis.argmax(dim=-1, keepdim=True)
        sign_vec = torch.gather(vee, -1, largest)
        sign = torch.where(sign_vec < 0.0, -torch.ones_like(sign_vec), torch.ones_like(sign_vec))
        axis = axis * sign
        pi_solution = axis * theta.unsqueeze(-1)
        regular = torch.where(near_pi.unsqueeze(-1), pi_solution, regular)

    return torch.where((theta < 1.0e-6).unsqueeze(-1), near_zero, regular)


def se3_exp(twist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按照旋量约定，将 ``xi = [omega, v]`` 指数映射为刚体变换。"""
    omega = twist[..., :3]
    v = twist[..., 3:]
    rotation = so3_exp(omega)
    omega_hat = skew(omega)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2.clamp_min(_EPS))
    eye = torch.eye(3, dtype=twist.dtype, device=twist.device).expand_as(omega_hat)
    a = torch.where(theta2 < 1.0e-10, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2)
    b = torch.where(
        theta2 < 1.0e-10,
        1.0 / 6.0 - theta2 / 120.0,
        (theta - torch.sin(theta)) / (theta2 * theta),
    )
    v_matrix = eye + a * omega_hat + b * (omega_hat @ omega_hat)
    translation = (v_matrix @ v.unsqueeze(-1)).squeeze(-1)
    return rotation, translation


def se3_log(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """按照旋量约定，将刚体变换对数映射为六维向量。"""
    omega = so3_log(rotation)
    omega_hat = skew(omega)
    theta2 = (omega * omega).sum(dim=-1, keepdim=True).unsqueeze(-1)
    theta = torch.sqrt(theta2.clamp_min(_EPS))
    eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device).expand_as(omega_hat)
    a = torch.where(theta2 < 1.0e-10, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / theta2)
    b = torch.where(
        theta2 < 1.0e-10,
        1.0 / 6.0 - theta2 / 120.0,
        (theta - torch.sin(theta)) / (theta2 * theta),
    )
    v_matrix = eye + a * omega_hat + b * (omega_hat @ omega_hat)
    v = torch.linalg.solve(v_matrix, translation.unsqueeze(-1)).squeeze(-1)
    return torch.cat((omega, v), dim=-1)


def transform_points(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """沿点张量的前导维度应用刚体变换。"""
    leading = torch.broadcast_shapes(points.shape[:-2], rotation.shape[:-2], translation.shape[:-1])
    if points.shape[:-2] != leading:
        points = points.expand(*leading, *points.shape[-2:])
    if rotation.shape[:-2] != leading:
        rotation = rotation.expand(*leading, 3, 3)
    if translation.shape[:-1] != leading:
        translation = translation.expand(*leading, 3)
    return points @ rotation.transpose(-1, -2) + translation.unsqueeze(-2)


def apply_object_action(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    centroids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """围绕每个物体质心应用物体级位姿增量。

    ``rotation`` 表示世界坐标中的姿态增量，``translation`` 表示质心位移。
    旋转不以世界坐标原点为中心。
    """
    if centroids is None:
        centroids = points.mean(dim=-2)
    return (
        points - centroids.unsqueeze(-2)
    ) @ rotation.transpose(-1, -2) + centroids.unsqueeze(-2) + translation.unsqueeze(-2)


def invert_rigid(rotation: torch.Tensor, translation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_inv = rotation.transpose(-1, -2)
    translation_inv = -(translation.unsqueeze(-2) @ rotation_inv).squeeze(-2)
    return rotation_inv, translation_inv


def compose_rigid(
    rotation_a: torch.Tensor,
    translation_a: torch.Tensor,
    rotation_b: torch.Tensor,
    translation_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """按先应用 A、再应用 B 的顺序组合刚体变换。"""
    rotation = rotation_b @ rotation_a
    translation = (
        translation_a.unsqueeze(-2) @ rotation_b.transpose(-1, -2)
    ).squeeze(-2) + translation_b
    return rotation, translation


def rotation_geodesic_loss(rotation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = rotation.transpose(-1, -2) @ target
    return so3_log(relative).square().sum(dim=-1)


def _deterministic_frame(points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """返回物体质心和右手 PCA 坐标系。

    坐标轴符号由三阶矩确定。对称物体可能出现奇异值相等，此时仍通过力矩
    符号和右手系补全保持确定性，仅精确退化情形需要特殊处理。
    """
    if mask is not None:
        if mask.ndim == points.ndim - 2:
            mask = mask.unsqueeze(-1).expand(*mask.shape, points.shape[-2])
        if mask.shape != points.shape[:-1]:
            raise ValueError(
                f"Point mask must be {tuple(points.shape[:-2]) + (points.shape[-2],)}, "
                f"got {tuple(mask.shape)}"
            )
        weights = mask.to(points.dtype)[..., None]
        denom = weights.sum(dim=-2).clamp_min(1.0)
        centroid = (points * weights).sum(dim=-2) / denom
    else:
        centroid = points.mean(dim=-2)

    centered = points - centroid.unsqueeze(-2)
    if mask is not None:
        centered = centered * mask.to(points.dtype).unsqueeze(-1)
    covariance = centered.transpose(-1, -2) @ centered
    covariance = covariance / max(points.shape[-2], 1)
    # 对称矩阵特征分解适用于小规模局部点集。
    _, vecs = torch.linalg.eigh(covariance)
    frame = vecs.flip(-1).transpose(-1, -2)  # 每一行表示一个主轴方向

    axis0 = frame[..., 0, :]
    axis1 = frame[..., 1, :]
    moment0 = (centered.square().sum(dim=-1, keepdim=True).sqrt() * (centered @ axis0.unsqueeze(-1))).sum(dim=-2).squeeze(-1)
    moment1 = (centered.square().sum(dim=-1, keepdim=True).sqrt() * (centered @ axis1.unsqueeze(-1))).sum(dim=-2).squeeze(-1)
    frame[..., 0, :] = torch.where((moment0 < 0.0).unsqueeze(-1), -axis0, axis0)
    frame[..., 1, :] = torch.where((moment1 < 0.0).unsqueeze(-1), -axis1, axis1)
    frame[..., 2, :] = torch.cross(frame[..., 0, :], frame[..., 1, :], dim=-1)
    return centroid, frame


@dataclass
class CanonicalPointCloud:
    """包含显式世界坐标到局部坐标变换的物体中心表示。"""

    points: torch.Tensor
    centroid: torch.Tensor
    frame: torch.Tensor

    def to_world(self, local_points: torch.Tensor) -> torch.Tensor:
        return self.centroid.unsqueeze(-2) + local_points @ self.frame

    def to_local(self, world_points: torch.Tensor) -> torch.Tensor:
        return (world_points - self.centroid.unsqueeze(-2)) @ self.frame.transpose(-1, -2)


def canonicalize_objects(points: torch.Tensor, object_mask: Optional[torch.Tensor] = None) -> CanonicalPointCloud:
    """将 ``(..., O, N, 3)`` 点张量中的每个物体转换到 PCA 规范坐标。"""
    if points.ndim < 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected (..., O, N, 3), got {tuple(points.shape)}")
    centroid, frame = _deterministic_frame(points, object_mask)
    local = (points - centroid.unsqueeze(-2)) @ frame.transpose(-1, -2)
    return CanonicalPointCloud(local, centroid, frame)


def world_action_to_local(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将世界坐标刚体增量转换为物体局部坐标。"""
    local_rotation = frame @ rotation @ frame.transpose(-1, -2)
    local_translation = (translation.unsqueeze(-2) @ frame.transpose(-1, -2)).squeeze(-2)
    return local_rotation, local_translation


def local_action_to_world(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    frame: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将物体局部坐标刚体增量提升到世界坐标。"""
    world_rotation = frame.transpose(-1, -2) @ rotation @ frame
    world_translation = (translation.unsqueeze(-2) @ frame).squeeze(-2)
    return world_rotation, world_translation


def action_6d_to_rt(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """将 ``(..., 6)`` 旋量解码为旋转矩阵和平移向量。"""
    if action.shape[-1] != 6:
        raise ValueError(f"Expected last dimension 6, got {action.shape[-1]}")
    return se3_exp(action)


def rt_to_action_6d(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return se3_log(rotation, translation)


def estimate_rigid_transform(source: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """根据已知点对应关系估计最小二乘刚体变换。

    源点和目标点的形状均为 ``(..., N, 3)``，返回由源点到目标点的变换。
    对含噪点轨迹使用重新正交化的 SVD 保证旋转矩阵合法。
    """
    if source.shape != target.shape:
        raise ValueError(f"Correspondences must have equal shapes, got {source.shape} and {target.shape}")
    if mask is not None:
        weights = mask.to(source.dtype)[..., None]
        denom = weights.sum(dim=-2).clamp_min(1.0)
        source_centroid = (source * weights).sum(dim=-2) / denom
        target_centroid = (target * weights).sum(dim=-2) / denom
        source_centered = (source - source_centroid.unsqueeze(-2)) * weights
        target_centered = (target - target_centroid.unsqueeze(-2)) * weights
    else:
        source_centroid = source.mean(dim=-2)
        target_centroid = target.mean(dim=-2)
        source_centered = source - source_centroid.unsqueeze(-2)
        target_centered = target - target_centroid.unsqueeze(-2)

    covariance = source_centered.transpose(-1, -2) @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.transpose(-1, -2) @ u.transpose(-1, -2)
    det = torch.linalg.det(rotation)
    correction = torch.eye(
        3,
        dtype=rotation.dtype,
        device=rotation.device,
    ).expand_as(rotation).clone()
    correction[..., -1, -1] = torch.where(det < 0.0, -torch.ones_like(det), torch.ones_like(det))
    rotation = rotation @ correction
    translation = target_centroid - (source_centroid.unsqueeze(-2) @ rotation.transpose(-1, -2)).squeeze(-2)
    return rotation, translation


def estimate_object_action(
    source: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """估计世界坐标中的姿态变化和质心位移。"""
    rotation, _ = estimate_rigid_transform(source, target, mask)
    if mask is not None:
        weights = mask.to(source.dtype)[..., None]
        denom = weights.sum(dim=-2).clamp_min(1.0)
        source_centroid = (source * weights).sum(dim=-2) / denom
        target_centroid = (target * weights).sum(dim=-2) / denom
    else:
        source_centroid = source.mean(dim=-2)
        target_centroid = target.mean(dim=-2)
    return rotation, target_centroid - source_centroid


def interpolate_rigid(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    fractions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从单位变换插值到目标刚体变换 ``(R, t)``。

    ``fractions`` 会广播到变换的前导维度。旋转沿最短测地线插值，
    平移采用线性插值。
    """
    omega = so3_log(rotation)
    scaled_omega = omega.unsqueeze(-2) * fractions[..., None]
    interpolated_rotation = so3_exp(scaled_omega)
    interpolated_translation = translation.unsqueeze(-2) * fractions[..., None]
    # 临时时间轴位于向量维度之前。
    # 将其移动到物体维度之前，使公开张量统一采用
    # “时域、物体、点、坐标”的维度顺序。
    interpolated_rotation = interpolated_rotation.transpose(-3, -4)
    interpolated_translation = interpolated_translation.transpose(-2, -3)
    return interpolated_rotation, interpolated_translation


def interpolate_trajectory_actions(
    total_rotation: torch.Tensor,
    total_translation: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将一个总刚体动作展开为逐步增量变换。"""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    fractions = torch.arange(1, horizon + 1, dtype=total_rotation.dtype, device=total_rotation.device)
    fractions = fractions / float(horizon)
    pose_rot, pose_trans = interpolate_rigid(total_rotation, total_translation, fractions)
    # 刚体插值函数返回“时域、物体、点、坐标”形状。
    identity_r = torch.eye(3, dtype=pose_rot.dtype, device=pose_rot.device).expand_as(pose_rot[:, 0])
    identity_t = torch.zeros_like(pose_trans[:, 0])
    previous_r = torch.cat((identity_r[:, None], pose_rot[:, :-1]), dim=1)
    previous_t = torch.cat((identity_t[:, None], pose_trans[:, :-1]), dim=1)
    # 物体动作围绕当前质心旋转，因此连续姿态增量相乘，质心位移相加。
    incremental_rotation = pose_rot @ previous_r.transpose(-1, -2)
    incremental_translation = pose_trans - previous_t
    return incremental_rotation, incremental_translation


def interpolate_keyframe_trajectory(
    keyframe_rotation: torch.Tensor,
    keyframe_translation: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 ``K`` 个累计物体位姿展开为 ``H`` 个增量命令。

    参数：
        keyframe_rotation：``(..., K, 3, 3)`` 相对当前物体位姿的累计旋转。
        keyframe_translation：``(..., K, 3)`` 累计质心偏移。
        horizon：可执行动作步数。

    轨迹沿最短路径 SO(3) 和线性质心插值通过每个关键帧。
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if keyframe_rotation.shape[-3] != keyframe_translation.shape[-2]:
        raise ValueError("rotation and translation keyframe counts must match")
    keyframes = keyframe_rotation.shape[-3]
    if keyframes < 1:
        raise ValueError("at least one keyframe is required")

    fractions = torch.arange(
        1,
        horizon + 1,
        dtype=keyframe_rotation.dtype,
        device=keyframe_rotation.device,
    ) / float(horizon)
    scaled = fractions * keyframes
    segment = torch.ceil(scaled).long().sub(1).clamp_(0, keyframes - 1)
    local_fraction = scaled - segment.to(scaled.dtype)
    local_fraction = torch.where(
        local_fraction <= 0.0,
        torch.ones_like(local_fraction),
        local_fraction,
    )

    def gather_keyframes(values: torch.Tensor, axis: int) -> torch.Tensor:
        axis = axis % values.ndim
        target_shape = list(values.shape)
        target_shape[axis] = horizon
        index = segment.view(
            *([1] * axis),
            horizon,
            *([1] * (values.ndim - axis - 1)),
        ).expand(*target_shape)
        return torch.gather(values, axis, index)

    current_rotation = gather_keyframes(keyframe_rotation, -3)
    current_translation = gather_keyframes(keyframe_translation, -2)
    previous_index = (segment - 1).clamp_min(-1)
    identity_rotation = torch.eye(
        3,
        dtype=keyframe_rotation.dtype,
        device=keyframe_rotation.device,
    ).expand(
        *keyframe_rotation.shape[:-3],
        horizon,
        3,
        3,
    ).clone()
    zero_translation = torch.zeros(
        *keyframe_translation.shape[:-2],
        horizon,
        3,
        dtype=keyframe_translation.dtype,
        device=keyframe_translation.device,
    )
    gather_previous = previous_index >= 0
    safe_previous = previous_index.clamp_min(0)

    def gather_previous_keyframes(values: torch.Tensor, axis: int) -> torch.Tensor:
        axis = axis % values.ndim
        target_shape = list(values.shape)
        target_shape[axis] = horizon
        index = safe_previous.view(
            *([1] * axis),
            horizon,
            *([1] * (values.ndim - axis - 1)),
        ).expand(*target_shape)
        return torch.gather(values, axis, index)

    previous_rotation = gather_previous_keyframes(keyframe_rotation, -3)
    previous_translation = gather_previous_keyframes(keyframe_translation, -2)
    rotation_mask = gather_previous.view(
        *([1] * len(keyframe_rotation.shape[:-3])),
        horizon,
        1,
        1,
    )
    translation_mask = gather_previous.view(
        *([1] * len(keyframe_translation.shape[:-2])),
        horizon,
        1,
    )
    previous_rotation = torch.where(
        rotation_mask,
        previous_rotation,
        identity_rotation,
    )
    previous_translation = torch.where(
        translation_mask,
        previous_translation,
        zero_translation,
    )

    relative_rotation = current_rotation @ previous_rotation.transpose(-1, -2)
    omega = so3_log(relative_rotation)
    pose_rotation = so3_exp(
        omega * local_fraction.view(
            *([1] * len(keyframe_rotation.shape[:-3])),
            horizon,
            1,
        )
    ) @ previous_rotation
    pose_translation = previous_translation + local_fraction.view(
        *([1] * len(keyframe_translation.shape[:-2])),
        horizon,
        1,
    ) * (current_translation - previous_translation)

    identity_start = torch.eye(
        3,
        dtype=pose_rotation.dtype,
        device=pose_rotation.device,
    ).expand_as(pose_rotation[..., 0, :, :])
    zero_start = torch.zeros_like(pose_translation[..., 0, :])
    previous_pose_rotation = torch.cat(
        (identity_start.unsqueeze(-3), pose_rotation[..., :-1, :, :]),
        dim=-3,
    )
    previous_pose_translation = torch.cat(
        (zero_start.unsqueeze(-2), pose_translation[..., :-1, :]),
        dim=-2,
    )
    incremental_rotation = (
        pose_rotation @ previous_pose_rotation.transpose(-1, -2)
    )
    incremental_translation = pose_translation - previous_pose_translation
    incremental_rotation = incremental_rotation.transpose(-4, -3)
    incremental_translation = incremental_translation.transpose(-2, -3)
    return incremental_rotation, incremental_translation


def apply_point_mask(points: torch.Tensor, mask: Optional[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
    if mask is None:
        return points
    return points * mask.to(points.dtype).unsqueeze(-1)


def farthest_point_indices(points: torch.Tensor, num_points: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """对 ``(..., N, 3)`` 点集执行确定性最远点采样。"""
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected (..., N, 3), got {tuple(points.shape)}")
    n = points.shape[-2]
    if n == 0 or num_points <= 0:
        raise ValueError("points and num_points must be non-empty")
    num_points = min(num_points, n)
    leading = points.shape[:-2]
    flat = points.reshape(-1, n, 3)
    batch = flat.shape[0]
    selected = torch.zeros(batch, num_points, dtype=torch.long, device=points.device)
    if generator is None:
        first = torch.zeros(batch, dtype=torch.long, device=points.device)
    else:
        first = torch.randint(0, n, (batch,), generator=generator, device=points.device)
    selected[:, 0] = first
    min_dist = torch.full((batch, n), float("inf"), dtype=points.dtype, device=points.device)
    current = first
    for i in range(1, num_points):
        current_points = flat[torch.arange(batch, device=points.device), current]
        dist = (flat - current_points.unsqueeze(-2)).square().sum(dim=-1)
        min_dist = torch.minimum(min_dist, dist)
        current = min_dist.argmax(dim=-1)
        selected[:, i] = current
    return selected.reshape(*leading, num_points)


def gather_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """沿倒数第二维点索引收集点坐标或逐点特征。"""
    expanded = indices.unsqueeze(-1).expand(*indices.shape, points.shape[-1])
    return torch.gather(points, -2, expanded)


def resample_points(points: torch.Tensor, num_points: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    indices = farthest_point_indices(points, num_points, generator=generator)
    return gather_points(points, indices)


def nearest_point_correspondence(
    query_points: torch.Tensor,
    source_points: torch.Tensor,
    target_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿源点云到目标点云的轨迹传播固定查询点编号。

    每个查询点先寻找源点云中的最近点，再返回目标点云中相同索引的点。
    当分割掩码逐帧轻微变化但 RGB-D 点云保持稳定像素索引时，该步骤用于
    维持关键物体点的固定对应关系。
    """
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have equal shapes")
    flat_source = source_points.reshape(-1, *source_points.shape[-2:])
    flat_target = target_points.reshape(-1, *target_points.shape[-2:])
    flat_query = query_points.reshape(-1, *query_points.shape[-2:])
    dist = torch.cdist(flat_query, flat_source)
    index = dist.argmin(dim=-1)
    gathered = torch.gather(flat_target, -2, index.unsqueeze(-1).expand_as(flat_query))
    return gathered.reshape_as(query_points), index.reshape_as(query_points[..., 0])


def pairwise_distance_loss(predicted: torch.Tensor, target: torch.Tensor, sample_limit: Optional[int] = 64) -> torch.Tensor:
    """计算近似刚体目标的点间距离一致性损失。"""
    n = predicted.shape[-2]
    if sample_limit is not None and n > sample_limit:
        idx = torch.linspace(0, n - 1, sample_limit, device=predicted.device).long()
        predicted = predicted[..., idx, :]
        target = target[..., idx, :]
    pred_dist = torch.cdist(predicted, predicted)
    target_dist = torch.cdist(target, target)
    return F.smooth_l1_loss(pred_dist, target_dist)


def point_cloud_bounds(points: Iterable[torch.Tensor], eps: float = 1.0e-3) -> tuple[torch.Tensor, torch.Tensor]:
    tensors = list(points)
    if not tensors:
        raise ValueError("At least one point tensor is required")
    flat = torch.cat([x.reshape(-1, 3) for x in tensors], dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(eps)
    return mean, std


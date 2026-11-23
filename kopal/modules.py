"""KOPAL 点云编码器和条件生成器的基础神经网络模块。

本文件实现层次点云编码、特征传播、正弦时间编码、显式多头注意力和条件
Transformer 块，供世界模型与底层策略共同使用。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig
from .geometry import farthest_point_indices, gather_points


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    if layers == 1:
        return nn.Sequential(nn.Linear(in_dim, out_dim))
    modules: list[nn.Module] = []
    current = in_dim
    for _ in range(layers - 1):
        modules.extend((nn.Linear(current, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()))
        current = hidden_dim
    modules.append(nn.Linear(current, out_dim))
    return nn.Sequential(*modules)


class SetAbstraction(nn.Module):
    """基于最近邻邻域的层次点云集合抽象层。"""

    def __init__(self, in_channels: int, out_channels: int, num_samples: int, knn: int) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.knn = knn
        self.local = nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.shape[:-2] != features.shape[:-2] or points.shape[-2] != features.shape[-2]:
            raise ValueError("points and features must share all but the channel dimension")
        b, n, _ = points.shape
        sample_n = min(self.num_samples, n)
        sample_indices = farthest_point_indices(points, sample_n)
        sampled_points = gather_points(points, sample_indices)
        k = min(self.knn, n)
        distance = torch.cdist(sampled_points, points)
        neighbor_indices = distance.topk(k, dim=-1, largest=False).indices
        flat = b * sample_n
        grouped_points = torch.gather(
            points.unsqueeze(1).expand(-1, sample_n, -1, -1),
            -2,
            neighbor_indices.reshape(b, sample_n, k, 1).expand(-1, -1, -1, 3),
        )
        grouped_features = torch.gather(
            features.unsqueeze(1).expand(-1, sample_n, -1, -1),
            -2,
            neighbor_indices.reshape(b, sample_n, k, 1).expand(-1, -1, -1, features.shape[-1]),
        )
        relative = grouped_points - sampled_points.unsqueeze(-2)
        local_input = torch.cat((relative, grouped_features), dim=-1)
        encoded = self.local(local_input).amax(dim=-2)
        return sampled_points, encoded.reshape(b, sample_n, -1)


class FeaturePropagation(nn.Module):
    """将稀疏点特征插值回稠密点云并融合跳连特征。"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + skip_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.SiLU(),
        )

    def forward(
        self,
        dense_points: torch.Tensor,
        sparse_points: torch.Tensor,
        sparse_features: torch.Tensor,
        skip_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if sparse_features.shape[-2] == 0:
            raise ValueError("sparse features cannot be empty")
        k = min(3, sparse_points.shape[-2])
        distance = torch.cdist(dense_points, sparse_points)
        weight, index = distance.topk(k, dim=-1, largest=False)
        weight = 1.0 / weight.clamp_min(1.0e-6)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        gathered = torch.gather(
            sparse_features.unsqueeze(1).expand(-1, dense_points.shape[-2], -1, -1),
            2,
            index.unsqueeze(-1).expand(-1, -1, -1, sparse_features.shape[-1]),
        )
        interpolated = (gathered * weight.unsqueeze(-1)).sum(dim=-2)
        if skip_features is None:
            skip_features = torch.zeros_like(interpolated[..., :0])
        fused = torch.cat((interpolated, skip_features), dim=-1)
        return self.mlp(fused)


class PointNetGlobalEncoder(nn.Module):
    """仅输入 XYZ 的逐点 MLP 与全局最大池化编码器。"""

    def __init__(self, out_dim: int = 256, hidden_dims: tuple[int, ...] = (64, 128, 256)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = 3
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            current = hidden
        layers.append(nn.Linear(current, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.net(points)
        if point_mask is not None:
            features = features.masked_fill(~point_mask.unsqueeze(-1), torch.finfo(features.dtype).min)
        return features.max(dim=-2).values


class PointNet2Encoder(nn.Module):
    """输出逐点特征和全局特征的轻量层次点云编码器。"""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        if len(config.point_hidden_dims) < 2:
            raise ValueError("point_hidden_dims needs at least two levels")
        c1, c2 = config.point_hidden_dims[:2]
        a1 = min(config.abstract_points[0], 1024) if config.abstract_points else 32
        a2 = min(config.abstract_points[1], a1) if len(config.abstract_points) > 1 else max(a1 // 2, 1)
        self.sa1 = SetAbstraction(3, c1, a1, config.knn)
        self.sa2 = SetAbstraction(c1, c2, a2, config.knn)
        self.fp2 = FeaturePropagation(c2, c1, c1)
        self.fp1 = FeaturePropagation(c1, 3, config.token_dim)
        self.global_head = mlp(config.token_dim, config.global_dim, config.global_dim, layers=2)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim < 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected (..., N, 3), got {tuple(points.shape)}")
        leading_shape = points.shape[:-2]
        n = points.shape[-2]
        flat_points = points.reshape(-1, n, 3)
        flat_mask = point_mask.reshape(-1, n) if point_mask is not None else None

        xyz1, feat1 = self.sa1(flat_points, flat_points)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        up1 = self.fp2(xyz1, xyz2, feat2, feat1)
        per_point = self.fp1(flat_points, xyz1, up1, flat_points)
        if flat_mask is not None:
            per_point = per_point * flat_mask.to(per_point.dtype).unsqueeze(-1)
            valid_object = flat_mask.any(dim=-1)
            global_input = per_point.masked_fill(
                ~flat_mask.unsqueeze(-1),
                torch.finfo(per_point.dtype).min,
            )
            # 全部点均无效的物体不能将 dtype 最小值送入 LayerNorm，
            # 对此类物体显式使用有限的零向量。
            global_input = global_input.masked_fill(
                ~valid_object.unsqueeze(-1).unsqueeze(-1),
                0.0,
            )
            global_feature = global_input.amax(dim=-2)
        else:
            valid_object = torch.ones(
                flat_points.shape[0],
                dtype=torch.bool,
                device=flat_points.device,
            )
            global_feature = per_point.amax(dim=-2)
        global_feature = self.global_head(global_feature)
        global_feature = global_feature * valid_object.to(
            global_feature.dtype
        ).unsqueeze(-1)
        return (
            per_point.reshape(*leading_shape, n, -1),
            global_feature.reshape(*leading_shape, -1),
        )


def sinusoidal_time_embedding(time: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("time embedding dimension must be even")
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=time.dtype, device=time.device)
        / max(half - 1, 1)
    )
    phase = time.unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ManualMultiheadAttention(nn.Module):
    """显式矩阵乘法实现的多头注意力。

    MeanFlow 需要在速度网络上计算 Jacobian-vector product。为避免部分
    PyTorch 版本的融合注意力内核不支持前向自动微分，本实现使用显式的
    matmul 和 softmax 计算注意力。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch, query_len, _ = query.shape
        key_len = key.shape[1]

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(batch, x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(query))
        k = split_heads(self.k_proj(key))
        v = split_heads(self.v_proj(value))
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool),
                torch.finfo(scores.dtype).min,
            )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = weights @ v
        attended = attended.transpose(1, 2).reshape(batch, query_len, self.d_model)
        output = self.out_proj(attended)
        return output, weights if need_weights else None


class ConditionalTransformerBlock(nn.Module):
    """包含自注意力、条件交叉注意力和前馈网络的 Transformer 块。"""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = ManualMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.cond_norm_q = nn.LayerNorm(d_model)
        self.cond_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = ManualMultiheadAttention(d_model, num_heads, dropout=dropout)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        condition_mask: Optional[torch.Tensor] = None,
        self_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.self_norm(x)
        self_out = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=self_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(self_out)
        cross_out = self.cross_attn(
            self.cond_norm_q(x),
            self.cond_norm_kv(condition),
            self.cond_norm_kv(condition),
            key_padding_mask=condition_mask,
            need_weights=False,
        )[0]
        x = x + self.dropout(cross_out)
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor], beta: float = 0.05) -> torch.Tensor:
    loss = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").mean(dim=-1)
    if mask is None:
        return loss.mean()
    weight = mask.to(loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1.0)


class SketchIsotropicGaussianRegularizer(nn.Module):
    """通过随机投影约束潜变量分布的各向同性高斯正则项。

    该模块在随机一维投影上计算 Epps-Pulley 特征函数统计量，使 embedding
    的分布接近各向同性标准高斯分布。与仅约束逐维方差相比，该统计量还能
    识别非高斯分布和低秩塌缩。
    """

    def __init__(self, knots: int = 17, num_proj: int = 256) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / max(knots - 1, 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", weights * window, persistent=False)

    def forward(self, projected_embeddings: torch.Tensor) -> torch.Tensor:
        """计算形状为 ``(样本组, 样本数, 维度)`` 的 embedding 正则损失。"""
        projection = torch.randn(
            projected_embeddings.shape[-1],
            self.num_proj,
            device=projected_embeddings.device,
            dtype=projected_embeddings.dtype,
        )
        projection = projection.div_(projection.norm(p=2, dim=0).clamp_min(1.0e-8))
        projected = projected_embeddings @ projection
        projected = projected.unsqueeze(-1) * self.t.to(projected.dtype)
        residual = (
            (projected.cos().mean(dim=-3) - self.phi.to(projected.dtype)).square()
            + projected.sin().mean(dim=-3).square()
        )
        statistic = (residual @ self.weights.to(projected.dtype)) * projected_embeddings.size(-2)
        return statistic.mean()


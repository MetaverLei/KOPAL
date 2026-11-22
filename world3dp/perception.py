"""KOPAL 的 RGB-D 融合与关键物体点云提取。

本文件实现深度反投影、多视角点云融合、Grounding DINO 文本检测、SAM2
实例分割和跨帧固定点对应维护。模型核心只依赖固定数据格式；外部检测器
权重不可用时，可以直接输入预计算掩码和点云。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol, Sequence

import torch
import torch.nn.functional as F

from .config import PerceptionConfig
from .geometry import canonicalize_objects, farthest_point_indices, gather_points, nearest_point_correspondence, resample_points

try:
    import numpy as np
except ImportError:
    np = None


@dataclass
class RGBDFrame:
    rgb: Any
    depth: Any
    intrinsics: Any
    camera_to_world: Any

    def __post_init__(self) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for RGBDFrame")
        self.rgb = np.asarray(self.rgb)
        self.depth = np.asarray(self.depth)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        self.camera_to_world = np.asarray(self.camera_to_world, dtype=np.float32)
        if self.depth.ndim != 2:
            raise ValueError("depth must be a 2D array")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be 3x3")
        if self.camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world must be 4x4")
        if self.rgb.shape[:2] != self.depth.shape:
            raise ValueError("RGB and depth spatial dimensions must match")


@dataclass
class Segmentation:
    label: str
    mask: Any
    score: float
    box: Optional[Any] = None


class Segmenter(Protocol):
    def segment(self, image: Any, text: str) -> Sequence[Segmentation]:
        """根据文本提示返回分割掩码。"""


class GroundedSAM2Segmenter:
    """Grounding DINO 与 SAM2 的惰性加载适配器。

    检测器依赖和权重在首次使用时加载，因此训练世界模型和策略不依赖视觉
    权重。``grounding_backend`` 和 ``sam2_backend`` 用于注入替代检测或
    分割后端，同时保持完整点云提取流程。
    """

    def __init__(
        self,
        model_name_or_path: str = "IDEA-Research/grounding-dino-tiny",
        sam2_checkpoint: Optional[str] = None,
        sam2_config: Optional[str] = None,
        device: str = "cpu",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        grounding_backend: Optional[Any] = None,
        sam2_backend: Optional[Any] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_config = sam2_config
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.grounding_backend = grounding_backend
        self.sam2_backend = sam2_backend
        self._processor = None
        self._grounding_model = None
        self._sam2_predictor = None

    def _load(self) -> None:
        if self._grounding_model is not None and self._sam2_predictor is not None:
            return
        if self.grounding_backend is not None and self.sam2_backend is not None:
            return
        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Grounding DINO requires `transformers`. Install it or inject "
                "`grounding_backend` and `sam2_backend`."
            ) from error
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as error:
            raise RuntimeError(
                "SAM2 requires the `sam2` package. Install it or inject a "
                "`sam2_backend`."
            ) from error
        if self.sam2_checkpoint is None or self.sam2_config is None:
            raise ValueError("sam2_checkpoint and sam2_config are required for real SAM2 inference")
        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self._grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name_or_path
        ).to(self.device)
        self._grounding_model.eval()
        self._sam2_predictor = SAM2ImagePredictor(
            build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
        )

    def _grounding_infer(self, image: Any, text: str) -> tuple[Any, Any, Sequence[str]]:
        if self.grounding_backend is not None:
            result = self.grounding_backend(image, text)
            if isinstance(result, Mapping):
                return result["boxes"], result["scores"], result["labels"]
            return result
        from PIL import Image

        pil_image = Image.fromarray(image) if not hasattr(image, "size") else image
        prompt = text.lower().strip()
        if not prompt.endswith("."):
            prompt += "."
        inputs = self._processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._grounding_model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
        )[0]
        return result["boxes"], result["scores"], result["labels"]

    def _sam_infer(self, image: Any, boxes: Any) -> tuple[Any, Any]:
        if self.sam2_backend is not None:
            result = self.sam2_backend(image, boxes)
            return result if isinstance(result, tuple) else (result, None)
        image_array = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
        self._sam2_predictor.set_image(image_array)
        masks, scores, _ = self._sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(boxes, dtype=np.float32),
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, scores

    @torch.inference_mode()
    def segment(self, image: Any, text: str) -> list[Segmentation]:
        self._load()
        boxes, box_scores, labels = self._grounding_infer(image, text)
        boxes_array = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
        scores_array = (
            box_scores.detach().cpu().numpy()
            if hasattr(box_scores, "detach")
            else np.asarray(box_scores)
        )
        if boxes_array.size == 0:
            return []
        masks, mask_scores = self._sam_infer(image, boxes_array)
        masks_array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
        if masks_array.ndim == 2:
            masks_array = masks_array[None]
        if mask_scores is None:
            mask_scores = scores_array
        mask_scores_array = (
            mask_scores.detach().cpu().numpy()
            if hasattr(mask_scores, "detach")
            else np.asarray(mask_scores)
        )
        output: list[Segmentation] = []
        for label, box, score, mask, mask_score in zip(
            labels,
            boxes_array,
            scores_array,
            masks_array,
            mask_scores_array.reshape(-1),
        ):
            normalized_label = str(label).lower().strip().rstrip(".")
            output.append(
                Segmentation(
                    label=normalized_label,
                    mask=mask.astype(bool),
                    score=float(max(float(score), float(mask_score))),
                    box=box,
                )
            )
        return output


@dataclass
class PerceptionOutput:
    scene_points: torch.Tensor
    object_points: torch.Tensor
    object_mask: torch.Tensor
    point_mask: torch.Tensor
    obstacle_points: torch.Tensor
    object_names: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def backproject_depth(
    depth: Any,
    intrinsics: Any,
    camera_to_world: Any,
    depth_scale: float = 1.0,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
) -> torch.Tensor:
    """将单帧 RGB-D 数据反投影为世界坐标点云，返回 ``(N,3)``。"""
    if np is None:
        raise RuntimeError("NumPy is required for depth back-projection")
    depth_array = np.asarray(depth, dtype=np.float32)
    intrinsics_array = np.asarray(intrinsics, dtype=np.float32)
    transform = np.asarray(camera_to_world, dtype=np.float32)
    valid = (
        np.isfinite(depth_array)
        & (depth_array >= min_depth / depth_scale)
        & (depth_array <= max_depth / depth_scale)
    )
    v, u = np.nonzero(valid)
    z = depth_array[v, u] * depth_scale
    fx, fy = intrinsics_array[0, 0], intrinsics_array[1, 1]
    cx, cy = intrinsics_array[0, 2], intrinsics_array[1, 2]
    x = (u.astype(np.float32) + 0.5 - cx) / fx * z
    y = (v.astype(np.float32) + 0.5 - cy) / fy * z
    camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    world_points = (camera_points @ transform.T)[:, :3]
    return torch.from_numpy(world_points.astype(np.float32))


def _fit_points(points: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] == 0:
        return points.new_zeros(count, 3), torch.zeros(count, dtype=torch.bool, device=points.device)
    if points.shape[0] >= count:
        sampled = resample_points(points.unsqueeze(0), count).squeeze(0)
        return sampled, torch.ones(count, dtype=torch.bool, device=points.device)
    repeats = (count + points.shape[0] - 1) // points.shape[0]
    sampled = points.repeat(repeats, 1)[:count]
    mask = torch.zeros(count, dtype=torch.bool, device=points.device)
    mask[: points.shape[0]] = True
    return sampled, mask


class MultiViewRGBDPerception:
    """融合多视角 RGB-D 点云并维护关键物体点的固定编号。"""

    def __init__(
        self,
        segmenter: Segmenter,
        config: Optional[PerceptionConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.segmenter = segmenter
        self.config = config or PerceptionConfig()
        self.device = device
        self._previous_object_points: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self._previous_object_points.clear()

    def _stabilize_points(self, name: str, points: torch.Tensor) -> torch.Tensor:
        count = self.config.object_points
        if name not in self._previous_object_points:
            sampled, _ = _fit_points(points, count)
            self._previous_object_points[name] = sampled.detach().clone()
            return sampled
        previous = self._previous_object_points[name]
        if previous.shape[0] != count:
            sampled, _ = _fit_points(points, count)
            self._previous_object_points[name] = sampled.detach().clone()
            return sampled
        if points.shape[0] == 0:
            return previous.clone()
        source = points
        if source.shape[0] < count:
            source = source.repeat((count + source.shape[0] - 1) // source.shape[0], 1)[:count]
        direct, direct_index = nearest_point_correspondence(
            previous.unsqueeze(0),
            source.unsqueeze(0),
            source.unsqueeze(0),
        )
        direct_distance = torch.cdist(previous.unsqueeze(0), source.unsqueeze(0))
        nearest_distance = torch.gather(
            direct_distance,
            -1,
            direct_index.unsqueeze(-1),
        )
        if nearest_distance.median() < 1.0e-4:
            self._previous_object_points[name] = direct.squeeze(0).detach().clone()
            return direct.squeeze(0)
        previous_local = canonicalize_objects(previous.unsqueeze(0)).points
        current_canonical = canonicalize_objects(source.unsqueeze(0))
        # 将上一帧物体变换到当前规范坐标系，
        # 再把每个稳定关键点关联到当前物体表面的最近点。
        predicted_current = current_canonical.to_world(previous_local)
        associated, _ = nearest_point_correspondence(
            predicted_current,
            source.unsqueeze(0),
            source.unsqueeze(0),
        )
        associated = associated.squeeze(0)
        self._previous_object_points[name] = associated.detach().clone()
        return associated

    def process(
        self,
        frames: Sequence[RGBDFrame],
        text: str,
        object_names: Optional[Sequence[str]] = None,
    ) -> PerceptionOutput:
        if not frames:
            raise ValueError("At least one RGB-D frame is required")
        requested = [name.lower().strip() for name in object_names] if object_names else None
        object_clouds: Dict[str, list[torch.Tensor]] = {}
        scene_clouds: list[torch.Tensor] = []
        background_clouds: list[torch.Tensor] = []
        detected_labels: set[str] = set()
        scores: Dict[str, list[float]] = {}

        for frame in frames:
            cloud = backproject_depth(
                frame.depth,
                frame.intrinsics,
                frame.camera_to_world,
                self.config.depth_scale,
                self.config.min_depth,
                self.config.max_depth,
            )
            scene_clouds.append(cloud)
            height, width = frame.depth.shape
            v, u = np.nonzero(
                np.isfinite(frame.depth)
                & (frame.depth >= self.config.min_depth / self.config.depth_scale)
                & (frame.depth <= self.config.max_depth / self.config.depth_scale)
            )
            # 按像素顺序重新计算相机点云，使物体掩码索引到相同点。
            # 每个有效像素恰好对应一个点云点。
            z = frame.depth[v, u].astype(np.float32) * self.config.depth_scale
            k = frame.intrinsics
            x = (u.astype(np.float32) + 0.5 - k[0, 2]) / k[0, 0] * z
            y = (v.astype(np.float32) + 0.5 - k[1, 2]) / k[1, 1] * z
            camera_points = np.stack((x, y, z, np.ones_like(z)), axis=-1)
            ordered_cloud = torch.from_numpy((camera_points @ frame.camera_to_world.T)[:, :3])
            pixel_to_cloud = -np.ones((height, width), dtype=np.int64)
            pixel_to_cloud[v, u] = np.arange(v.shape[0], dtype=np.int64)
            object_pixel_mask = np.zeros((height, width), dtype=bool)
            segmentations = self.segmenter.segment(frame.rgb, text)
            for segmentation in segmentations:
                label = segmentation.label.lower().strip().rstrip(".")
                if requested is not None and label not in requested:
                    continue
                mask = np.asarray(segmentation.mask)
                if mask.shape != (height, width):
                    raise ValueError(f"Segmentation mask shape {mask.shape} does not match depth {(height, width)}")
                selected_pixels = mask & (pixel_to_cloud >= 0)
                object_pixel_mask |= selected_pixels
                indices = pixel_to_cloud[selected_pixels]
                if indices.size == 0:
                    continue
                object_clouds.setdefault(label, []).append(ordered_cloud[indices])
                scores.setdefault(label, []).append(float(segmentation.score))
                detected_labels.add(label)
            background_pixels = (pixel_to_cloud >= 0) & ~object_pixel_mask
            background_indices = pixel_to_cloud[background_pixels]
            if background_indices.size:
                background_clouds.append(ordered_cloud[background_indices])

        names = list(requested) if requested is not None else sorted(detected_labels)
        object_points = []
        object_valid = []
        point_masks = []
        for name in names:
            if name in object_clouds:
                cloud = torch.cat(object_clouds[name], dim=0)
                sampled = self._stabilize_points(name, cloud.to(self.device))
                valid = True
            else:
                sampled = torch.zeros(self.config.object_points, 3, device=self.device)
                valid = False
            mask = torch.ones(self.config.object_points, dtype=torch.bool, device=self.device) if valid else torch.zeros(
                self.config.object_points, dtype=torch.bool, device=self.device
            )
            object_points.append(sampled)
            object_valid.append(valid)
            point_masks.append(mask)

        if not object_points:
            object_tensor = torch.zeros(0, self.config.object_points, 3, device=self.device)
            object_mask = torch.zeros(0, dtype=torch.bool, device=self.device)
            point_mask = torch.zeros(0, self.config.object_points, dtype=torch.bool, device=self.device)
        else:
            object_tensor = torch.stack(object_points, dim=0)
            object_mask = torch.tensor(object_valid, dtype=torch.bool, device=self.device)
            point_mask = torch.stack(point_masks, dim=0)

        scene = torch.cat(scene_clouds, dim=0).to(self.device)
        scene_sampled, _ = _fit_points(scene, self.config.scene_points)
        background = torch.cat(background_clouds, dim=0).to(self.device) if background_clouds else scene.new_zeros(0, 3)
        if background.shape[0] > 0:
            obstacle, _ = _fit_points(background, min(self.config.scene_points, background.shape[0]))
        else:
            obstacle = background
        return PerceptionOutput(
            scene_points=scene_sampled,
            object_points=object_tensor,
            object_mask=object_mask,
            point_mask=point_mask,
            obstacle_points=obstacle,
            object_names=names,
            metadata={
                "scores": {name: max(values) if values else 0.0 for name, values in scores.items()},
            },
        )


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VMamba detection + ResNet ReID multi-object tracking for worker tracking.

Pipeline:
1. VMamba-based Mask R-CNN detects worker boxes.
2. ResNet-50 extracts visual appearance features from each detected box.
3. The tracker associates boxes across frames with IoU + center distance +
   ReID feature similarity, then writes MOT-format results.

Output format matches /home/lvxuan/llmtest/gdino_qwen25vl_reid_mot.py:
    frame_id,track_id,x1,y1,w,h,score,1,-1
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_MODULES_CACHE", "/data1/pengsihaoran/.cache/huggingface/modules")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

# Add VMamba detection to path - MUST be before mmdet imports
VMAMBA_ROOT = "/data1/pengsihaoran/mamba/VMamba-main"
DETECTION_ROOT = os.path.join(VMAMBA_ROOT, "detection")
CLASSIFICATION_ROOT = os.path.join(VMAMBA_ROOT, "classification")
sys.path.insert(0, DETECTION_ROOT)
sys.path.insert(0, CLASSIFICATION_ROOT)
sys.path.insert(0, VMAMBA_ROOT)

# Register VMamba backbone for MMDet 3.x
# Import VMamba detection model to register MM_VSSM
from mmdet.registry import MODELS

# Import VMamba backbone from classification models
sys.path.insert(0, os.path.join(CLASSIFICATION_ROOT, "models"))
try:
    from vmamba import Backbone_VSSM

    @MODELS.register_module(name='MM_VSSM')
    class MM_VSSM_Wrapper(Backbone_VSSM):
        """Wrapper for VMamba backbone compatible with MMDet 3.x"""
        def __init__(self, pretrained=None, out_indices=(0, 1, 2, 3), norm_layer="ln", **kwargs):
            super().__init__(out_indices=out_indices, norm_layer=norm_layer, **kwargs)
            if pretrained is not None and os.path.isfile(pretrained):
                print(f"Loading pretrained weights from {pretrained}")
                self.load_pretrained(pretrained)

except ImportError as e:
    print(f"Warning: Could not import VMamba classification models: {e}")
    Backbone_VSSM = None

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.ops import nms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as F

from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules as register_mmdet
register_mmdet()

DEFAULT_IMAGE_DIR = "/data1/pengsihaoran/data/dwtest/images"
DEFAULT_GT_ROOT = "/data1/pengsihaoran/data/dwtest/gt"
DEFAULT_OUTPUT_DIR = "/data1/pengsihaoran/mamba/outputs/vmamba_mot"
DEFAULT_VMAMBA_CONFIG = "/data1/pengsihaoran/mamba/VMamba-main/detection/configs/vssm1/mask_rcnn_vssm_fpn_coco_base.py"
DEFAULT_VMAMBA_CHECKPOINT = "/data1/pengsihaoran/mamba/VMamba-main/checkpoint/mask_rcnn_vssm_fpn_coco_base_epoch_11.pth"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 128, 128), (128, 255, 128),
]

def print_model_params(model, name: str = "Model") -> None:
    """Print model parameter statistics."""
    if model is None:
        print(f"{name}: NOT LOADED (None)", flush=True)
        return

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 60, flush=True)
    print(f"{name} - Parameters", flush=True)
    print("=" * 60, flush=True)
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)", flush=True)
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)", flush=True)
    print(f"Non-trainable Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e6:.2f}M)", flush=True)

    print("\nParameters by Module:", flush=True)
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {name}: {params:,} ({params/1e6:.2f}M)", flush=True)
    print("=" * 60 + "\n", flush=True)

def numeric_sort_key(path: str) -> Tuple[int, Any]:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


def frame_id_from_path(path: str, fallback: int) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return fallback


def resolve_split_dir(root_or_split_dir: str, split: str) -> str:
    split_dir = os.path.join(root_or_split_dir, split)
    if os.path.isdir(split_dir):
        return split_dir
    return root_or_split_dir


def list_image_files(image_dir: str) -> List[str]:
    if not os.path.isdir(image_dir):
        return []
    names = [
        name for name in os.listdir(image_dir)
        if os.path.splitext(name.lower())[1] in IMAGE_EXTS
    ]
    return [os.path.join(image_dir, name) for name in sorted(names, key=numeric_sort_key)]


def clip_box(box: Sequence[float], width: int, height: int) -> Optional[List[int]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def calculate_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    cx1, cy1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    cx2, cy2 = (x3 + x4) / 2.0, (y3 + y4) / 2.0
    return ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5


def box_diagonal(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def predict_box(box: Sequence[float], velocity: Sequence[float], age: int) -> List[float]:
    step = max(1, min(age, 3))
    return [float(value) + float(delta) * step for value, delta in zip(box, velocity)]


def expand_box(box: Sequence[float], width: int, height: int, pad_ratio: float) -> Optional[List[int]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    box_w = x2 - x1
    box_h = y2 - y1
    return clip_box(
        [
            x1 - box_w * pad_ratio,
            y1 - box_h * pad_ratio,
            x2 + box_w * pad_ratio,
            y2 + box_h * pad_ratio,
        ],
        width,
        height,
    )


def suppress_duplicate_detections(
    detections: Sequence[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    if iou_threshold <= 0 or len(detections) <= 1:
        return [dict(det) for det in detections]

    kept: List[Dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: item.get("score", 1.0), reverse=True):
        if any(calculate_iou(det["bbox"], kept_det["bbox"]) >= iou_threshold for kept_det in kept):
            continue
        kept.append(dict(det))
    return kept


class VMambaDetector:
    """VMamba-based detector using MMDetection framework."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        box_threshold: float,
        nms_iou: float,
        max_det: int,
        device: str = "cuda",
    ):
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print("Loading VMamba detector...")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")

        # Verify files exist before loading
        if not os.path.isfile(config_path):
            print(f"Error: Config file not found: {config_path}")
            self.model = None
            self.use_mmdet = False
        elif not os.path.isfile(checkpoint_path):
            print(f"Error: Checkpoint file not found: {checkpoint_path}")
            self.model = None
            self.use_mmdet = False
        else:
            try:
                self.model = init_detector(
                    config_path,
                    checkpoint_path,
                    device=self.device.type
                )
                if self.model is None:
                    print("Warning: init_detector returned None")
                    self.use_mmdet = False
                else:
                    self.use_mmdet = True
                    print(f"VMamba detector loaded on {self.device}.")
            except Exception as e:
                print(f"Warning: Failed to load VMamba detector: {e}")
                import traceback
                traceback.print_exc()
                self.model = None
                self.use_mmdet = False

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        height, width = image_bgr.shape[:2]

        if not self.use_mmdet or self.model is None:
            return []

        try:
            result = inference_detector(self.model, image_bgr)

            # Handle MMDet 3.x DetDataSample format
            from mmdet.structures import DetDataSample
            if isinstance(result, DetDataSample):
                pred_instances = result.pred_instances
                boxes = pred_instances.bboxes.cpu().numpy()
                scores = pred_instances.scores.cpu().numpy()
                labels = pred_instances.labels.cpu().numpy()
                print(f"  Debug - DetDataSample: {len(boxes)} detections, max score: {scores.max() if len(scores) > 0 else 0}")

                # Filter: only keep person class (COCO class_id=0)
                detections = []
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if label == 0 and score >= self.box_threshold:
                        x1, y1, x2, y2 = box
                        clipped = clip_box([x1, y1, x2, y2], width, height)
                        if clipped is None:
                            continue
                        detections.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "score": float(score),
                            "label": 0,
                        })
            elif isinstance(result, (tuple, list)):
                # Handle MMDet 2.x format: list of arrays per class
                detections = []
                for label, det in enumerate(result):
                    if det.shape[0] > 0:
                        for box in det:
                            x1, y1, x2, y2, score = box[:5]
                            # Only keep person class (COCO class_id=0)
                            if label == 0 and score >= self.box_threshold:
                                clipped = clip_box([x1, y1, x2, y2], width, height)
                                if clipped is None:
                                    continue
                                detections.append({
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "score": float(score),
                                    "label": label,
                                })
            else:
                detections = []

            # Apply NMS
            if len(detections) > 0:
                boxes_tensor = torch.tensor([d["bbox"] for d in detections], dtype=torch.float32)
                scores_tensor = torch.tensor([d["score"] for d in detections], dtype=torch.float32)
                keep = nms(boxes_tensor, scores_tensor, self.nms_iou)
                detections = [detections[i] for i in keep]

            detections.sort(key=lambda item: item["score"], reverse=True)
            if self.max_det > 0:
                detections = detections[:self.max_det]

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []


class ResNet50ReID:
    """ResNet-50 based ReID feature extractor."""

    def __init__(
        self,
        batch_size: int,
        crop_pad: float,
        dtype_name: str,
        reid_min_pixels: int,
        reid_max_pixels: int,
    ):
        self.batch_size = max(1, batch_size)
        self.crop_pad = crop_pad
        self.reid_min_pixels = max(0, reid_min_pixels)
        self.reid_max_pixels = max(0, reid_max_pixels)
        self.dtype = self._resolve_dtype(dtype_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading ResNet-50 for ReID...")
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.fc = nn.Identity()
        self.model = self.model.to(self.device)
        if self.dtype == torch.float16:
            self.model = self.model.half()
        self.model.eval()

        # Handle different weights metadata formats
        try:
            self.mean = torch.tensor(weights.meta["mean"]).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor(weights.meta["std"]).view(1, 3, 1, 1).to(self.device)
        except KeyError:
            # Default ImageNet stats
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        print(f"ResNet-50 ReID loaded on {self.device}.")

    @staticmethod
    def _resolve_dtype(dtype_name: str) -> torch.dtype:
        if dtype_name == "fp16":
            return torch.float16
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp32":
            return torch.float32
        return torch.float32

    def _crop_image(self, image_bgr: np.ndarray, box: Sequence[float]) -> Optional[np.ndarray]:
        height, width = image_bgr.shape[:2]
        crop_box = expand_box(box, width, height, self.crop_pad)
        if not crop_box:
            return None
        x1, y1, x2, y2 = crop_box
        patch = image_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        return patch

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        area = h * w

        if self.reid_min_pixels > 0 and area < self.reid_min_pixels:
            scale = np.sqrt(self.reid_min_pixels / area)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))
        elif self.reid_max_pixels > 0 and area > self.reid_max_pixels:
            scale = np.sqrt(self.reid_max_pixels / area)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h))

        return image

    def _extract_batch_features(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        if len(images) == 0:
            return torch.empty((0, 2048), dtype=torch.float32)

        # Resize all images to a fixed size for batch processing
        # Using 128x256 as standard ReID input size (aspect ratio friendly for person crops)
        target_h, target_w = 256, 128
        resized_images = []
        for img in images:
            img_resized = cv2.resize(img, (target_w, target_h))
            resized_images.append(img_resized)

        tensors = [F.to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in resized_images]
        batch_tensor = torch.stack(tensors).to(self.device)
        batch_tensor = (batch_tensor - self.mean) / self.std

        if self.dtype == torch.float16:
            batch_tensor = batch_tensor.half()

        with torch.inference_mode():
            features = self.model(batch_tensor)

        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1).cpu()
        return features

    def extract_batch_features(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        if not detections:
            return torch.empty((0, 0), dtype=torch.float32)

        patch_images = []
        valid_indices = []
        for idx, det in enumerate(detections):
            patch_image = self._crop_image(image_bgr, det["bbox"])
            if patch_image is None:
                continue
            patch_images.append(patch_image)
            valid_indices.append(idx)

        if not patch_images:
            return torch.zeros((len(detections), 1), dtype=torch.float32)

        features_by_det: List[Optional[torch.Tensor]] = [None] * len(detections)

        with torch.inference_mode():
            for start in range(0, len(patch_images), self.batch_size):
                end = start + self.batch_size
                features = self._extract_batch_features(patch_images[start:end])
                for local_idx, det_idx in enumerate(valid_indices[start:end]):
                    features_by_det[det_idx] = features[local_idx]

        feature_dim = next(item for item in features_by_det if item is not None).numel()
        zero = torch.zeros(feature_dim, dtype=torch.float32)
        return torch.stack([item if item is not None else zero for item in features_by_det])


class ReIDAssociator:
    """Multi-object tracker with ReID features."""

    def __init__(
        self,
        iou_weight: float,
        feature_weight: float,
        distance_weight: float,
        iou_threshold: float,
        feature_threshold: float,
        dist_threshold: float,
        match_threshold: float,
        max_age: int,
        feature_momentum: float,
        id_start: int,
        strong_iou_threshold: float,
        appearance_only_threshold: float,
        max_center_jump_ratio: float,
    ):
        self.iou_weight = iou_weight
        self.feature_weight = feature_weight
        self.distance_weight = distance_weight
        self.iou_threshold = iou_threshold
        self.feature_threshold = feature_threshold
        self.dist_threshold = dist_threshold
        self.match_threshold = match_threshold
        self.max_age = max_age
        self.feature_momentum = feature_momentum
        self.id_start = id_start
        self.strong_iou_threshold = strong_iou_threshold
        self.appearance_only_threshold = appearance_only_threshold
        self.max_center_jump_ratio = max_center_jump_ratio
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.next_id = id_start

    def reset(self) -> None:
        self.tracks = {}
        self.next_id = self.id_start

    def _feature_similarity(self, det_feature: torch.Tensor, track_feature: torch.Tensor) -> float:
        if det_feature.numel() == 0 or track_feature.numel() == 0:
            return 0.0
        if det_feature.numel() != track_feature.numel():
            return 0.0
        sim = float(torch.dot(det_feature.float(), track_feature.float()).item())
        if not np.isfinite(sim):
            return 0.0
        return sim

    def _distance_threshold(self, track: Dict[str, Any]) -> float:
        return max(
            self.dist_threshold,
            box_diagonal(track["bbox"]) * self.max_center_jump_ratio,
        )

    def _candidate_metrics(
        self,
        det: Dict[str, Any],
        det_feature: torch.Tensor,
        track: Dict[str, Any],
        frame_id: int,
    ) -> Dict[str, float]:
        age = frame_id - int(track["last_frame"])
        pred_box = predict_box(track["bbox"], track.get("velocity", [0.0, 0.0, 0.0, 0.0]), age)
        iou = calculate_iou(det["bbox"], pred_box)
        last_iou = calculate_iou(det["bbox"], track["bbox"])
        dist = center_distance(det["bbox"], pred_box)
        dist_threshold = self._distance_threshold(track)
        dist_score = max(0.0, 1.0 - dist / max(1.0, dist_threshold))
        sim = self._feature_similarity(det_feature, track["feature"])
        sim_score = float(np.clip((sim + 1.0) * 0.5, 0.0, 1.0))
        score = (
            self.iou_weight * iou
            + self.feature_weight * sim_score
            + self.distance_weight * dist_score
            + 0.05 * last_iou
            - 0.03 * max(0, age - 1)
        )
        return {
            "score": score,
            "iou": iou,
            "last_iou": last_iou,
            "sim": sim,
            "sim_score": sim_score,
            "dist": dist,
            "dist_score": dist_score,
            "dist_threshold": dist_threshold,
            "age": float(age),
        }

    def _assign_candidates(
        self,
        candidates: List[Tuple[float, int, int]],
        detections: List[Dict[str, Any]],
        used_dets: set,
        used_tracks: set,
    ) -> None:
        candidates.sort(reverse=True, key=lambda item: item[0])
        for _, det_idx, track_id in candidates:
            if det_idx in used_dets or track_id in used_tracks:
                continue
            detections[det_idx]["id"] = track_id
            used_dets.add(det_idx)
            used_tracks.add(track_id)

    def associate(
        self,
        detections: List[Dict[str, Any]],
        features: torch.Tensor,
        frame_id: int,
    ) -> List[Dict[str, Any]]:
        if len(detections) == 0:
            self._expire(frame_id)
            return []

        if features.numel() == 0:
            features = torch.zeros((len(detections), 1), dtype=torch.float32)

        used_dets = set()
        used_tracks = set()

        # Stage 1: preserve IDs for continuous motion
        geometry_candidates = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self.tracks.items():
                age = frame_id - int(track["last_frame"])
                if age <= 0 or age > self.max_age:
                    continue
                metrics = self._candidate_metrics(det, features[det_idx], track, frame_id)
                strong_iou = max(metrics["iou"], metrics["last_iou"])
                if age <= 2 and strong_iou >= self.strong_iou_threshold:
                    score = 2.0 + strong_iou + 0.25 * metrics["dist_score"]
                    geometry_candidates.append((score, det_idx, track_id))

        self._assign_candidates(geometry_candidates, detections, used_dets, used_tracks)

        # Stage 2: use geometry + ReID appearance
        reid_candidates = []
        for det_idx, det in enumerate(detections):
            if det_idx in used_dets:
                continue
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                age = frame_id - int(track["last_frame"])
                if age <= 0 or age > self.max_age:
                    continue
                metrics = self._candidate_metrics(det, features[det_idx], track, frame_id)
                spatial_match = (
                    metrics["iou"] >= self.iou_threshold
                    or metrics["last_iou"] >= self.iou_threshold
                    or metrics["dist"] <= metrics["dist_threshold"]
                )
                appearance_match = metrics["sim"] >= self.appearance_only_threshold
                if not spatial_match and not appearance_match:
                    continue
                if metrics["score"] >= self.match_threshold:
                    reid_candidates.append((metrics["score"], det_idx, track_id))

        self._assign_candidates(reid_candidates, detections, used_dets, used_tracks)

        # Assign new IDs to unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                det["id"] = self.next_id
                self.next_id += 1

        # Update tracks
        for det_idx, det in enumerate(detections):
            track_id = int(det["id"])
            new_feature = features[det_idx].float()
            if track_id in self.tracks:
                old_bbox = self.tracks[track_id]["bbox"]
                old_feature = self.tracks[track_id]["feature"]
                if old_feature.numel() == new_feature.numel():
                    smoothed = (
                        old_feature * self.feature_momentum
                        + new_feature * (1.0 - self.feature_momentum)
                    )
                    new_feature = torch.nn.functional.normalize(smoothed, p=2, dim=0)
                velocity = [float(det["bbox"][idx]) - float(old_bbox[idx]) for idx in range(4)]
                hits = int(self.tracks[track_id].get("hits", 0)) + 1
            else:
                velocity = [0.0, 0.0, 0.0, 0.0]
                hits = 1

            self.tracks[track_id] = {
                "bbox": det["bbox"],
                "feature": new_feature,
                "last_frame": frame_id,
                "velocity": velocity,
                "hits": hits,
            }

        self._expire(frame_id)
        return detections

    def _expire(self, frame_id: int) -> None:
        expired = [
            track_id for track_id, track in self.tracks.items()
            if frame_id - int(track["last_frame"]) > self.max_age
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)


class VMambaMOT:
    """Main MOT class combining VMamba detector and ReID tracker."""

    def __init__(self, args: argparse.Namespace):
        self.detector = VMambaDetector(
            config_path=args.vmamba_config,
            checkpoint_path=args.vmamba_checkpoint,
            box_threshold=args.box_threshold,
            nms_iou=args.det_nms_iou,
            max_det=args.max_det,
        )
        self.reid = ResNet50ReID(
            batch_size=args.reid_batch_size,
            crop_pad=args.reid_crop_pad,
            dtype_name=args.dtype,
            reid_min_pixels=args.reid_min_pixels,
            reid_max_pixels=args.reid_max_pixels,
        )
        self.associator = ReIDAssociator(
            iou_weight=args.iou_weight,
            feature_weight=args.feature_weight,
            distance_weight=args.distance_weight,
            iou_threshold=args.iou_threshold,
            feature_threshold=args.feature_threshold,
            dist_threshold=args.dist_threshold,
            match_threshold=args.match_threshold,
            max_age=args.max_age,
            feature_momentum=args.feature_momentum,
            id_start=args.id_start,
            strong_iou_threshold=args.strong_iou_threshold,
            appearance_only_threshold=args.appearance_only_threshold,
            max_center_jump_ratio=args.max_center_jump_ratio,
        )
        self.track_nms_iou = args.track_nms_iou
        self.show_stats = args.show_stats
        self.show_power = args.show_stats_power
        self.fps_list: List[float] = []
        self.peak_memory_gb = 0.0
        self.power_samples: List[float] = []
        self.peak_power_w = 0.0
        self.avg_power_w = 0.0
        self._nvml_initialized = False
        if self.show_power and PYNVML_AVAILABLE and torch.cuda.is_available():
            try:
                pynvml.nvmlInit()
                self.device_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_initialized = True
                print("GPU power monitoring initialized", flush=True)
            except Exception as e:
                print(f"Warning: Could not initialize NVML: {e}", flush=True)
                self._nvml_initialized = False

    def reset(self) -> None:
        self.associator.reset()
        self.fps_list = []
        self.peak_memory_gb = 0.0
        self.power_samples = []
        self.peak_power_w = 0.0
        self.avg_power_w = 0.0

    def track_frame(self, image_path: str, frame_id: int) -> List[Dict[str, Any]]:
        start = time.time()
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: failed to read image: {image_path}")
            return []

        if self.show_stats and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        detections = self.detector.detect(image_bgr)
        detections = suppress_duplicate_detections(detections, self.track_nms_iou)
        features = self.reid.extract_batch_features(image_bgr, detections)
        tracked = self.associator.associate(detections, features, frame_id)

        if self.show_stats:
            elapsed = time.time() - start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.fps_list.append(fps)
            if torch.cuda.is_available():
                mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                self.peak_memory_gb = max(self.peak_memory_gb, mem)
            print(
                f"Frame {frame_id}: {elapsed:.2f}s ({fps:.2f} FPS), "
                f"det={len(tracked)}, peak={self.peak_memory_gb:.2f} GB"
            )

        if self.show_power and self._nvml_initialized:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self.device_handle)
                power_w = power_mw / 1000.0
                self.power_samples.append(power_w)
                self.peak_power_w = max(self.peak_power_w, power_w)
                print(f"Frame {frame_id}: Power={power_w:.2f}W (peak={self.peak_power_w:.2f}W)")
            except Exception as e:
                print(f"Warning: Could not read power: {e}", flush=True)

        return tracked

    def print_stats(self) -> None:
        if not self.fps_list:
            return
        print("\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)
        print(f"Total frames: {len(self.fps_list)}")
        print(f"Peak GPU memory: {self.peak_memory_gb:.2f} GB")
        if hasattr(np, 'mean'):
            print(f"Average FPS: {np.mean(self.fps_list):.2f}")
            print(f"Median FPS: {np.median(self.fps_list):.2f}")
        print(f"Min FPS: {min(self.fps_list):.2f}")
        print(f"Max FPS: {max(self.fps_list):.2f}")
        print("=" * 60)

        if self.power_samples:
            print("\n" + "=" * 60)
            print("GPU Power Statistics")
            print("=" * 60)
            self.avg_power_w = sum(self.power_samples) / len(self.power_samples)
            print(f"Average Power: {self.avg_power_w:.2f} W")
            print(f"Peak Power: {self.peak_power_w:.2f} W")
            print(f"Min Power: {min(self.power_samples):.2f} W")
            print(f"Max Power: {max(self.power_samples):.2f} W")
            print(f"Power Samples: {len(self.power_samples)}")
            print("=" * 60)

        if self._nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self._nvml_initialized = False
            except Exception:
                pass


def write_mot_lines(handle: Any, frame_id: int, detections: Sequence[Dict[str, Any]]) -> None:
    """Write MOT format output - same format as gdino_qwen25vl_reid_mot.py"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        handle.write(
            f"{frame_id},{det['id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
            f"{det.get('score', 1.0):.4f},1,-1\n"
        )


def visualize_frame(image_path: str, detections: Sequence[Dict[str, Any]], output_path: str) -> None:
    """Visualize tracking results on frame."""
    image = cv2.imread(image_path)
    if image is None:
        return
    height, width = image.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        track_id = int(det["id"])
        color = COLORS[track_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{track_id} {det.get('score', 1.0):.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        label_top = max(0, y1 - text_h - baseline)
        cv2.rectangle(image, (x1, label_top), (x1 + text_w, y1), color, -1)
        cv2.putText(
            image,
            label,
            (x1, max(text_h, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VMamba-based Multi-Object Tracking")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--limit_seq", type=int, default=0)
    parser.add_argument("--limit_frames", type=int, default=0)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--show_stats", action="store_true")
    parser.add_argument("--show_stats_power", action="store_true", help="Show power consumption statistics (requires pynvml)")
    parser.add_argument("--show_params", action="store_true")

    parser.add_argument("--vmamba_config", type=str, default=DEFAULT_VMAMBA_CONFIG)
    parser.add_argument("--vmamba_checkpoint", type=str, default=DEFAULT_VMAMBA_CHECKPOINT)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--det_nms_iou", type=float, default=0.60)
    parser.add_argument("--track_nms_iou", type=float, default=0.55)
    parser.add_argument("--max_det", type=int, default=20)

    parser.add_argument("--reid_batch_size", type=int, default=4)
    parser.add_argument("--reid_crop_pad", type=float, default=0.08)
    parser.add_argument("--reid_min_pixels", type=int, default=3136)
    parser.add_argument("--reid_max_pixels", type=int, default=200704)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")

    parser.add_argument("--iou_weight", type=float, default=0.35)
    parser.add_argument("--feature_weight", type=float, default=0.45)
    parser.add_argument("--distance_weight", type=float, default=0.20)
    parser.add_argument("--iou_threshold", type=float, default=0.10)
    parser.add_argument("--feature_threshold", type=float, default=0.45)
    parser.add_argument("--dist_threshold", type=float, default=180.0)
    parser.add_argument("--match_threshold", type=float, default=0.30)
    parser.add_argument("--strong_iou_threshold", type=float, default=0.25)
    parser.add_argument("--appearance_only_threshold", type=float, default=0.75)
    parser.add_argument("--max_center_jump_ratio", type=float, default=0.80)
    parser.add_argument("--max_age", type=int, default=20)
    parser.add_argument("--feature_momentum", type=float, default=0.80)
    parser.add_argument("--id_start", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.vmamba_checkpoint):
        raise SystemExit(f"VMamba checkpoint not found: {args.vmamba_checkpoint}")
    if not os.path.isfile(args.vmamba_config):
        raise SystemExit(f"VMamba config not found: {args.vmamba_config}")

    image_split_dir = resolve_split_dir(args.image_dir, args.split)
    gt_split_dir = resolve_split_dir(args.gt_root, args.split)
    if not os.path.isdir(image_split_dir):
        raise SystemExit(f"Image directory not found: {image_split_dir}")

    seq_list = sorted(
        name for name in os.listdir(image_split_dir)
        if os.path.isdir(os.path.join(image_split_dir, name))
    )
    if args.seq_name:
        seq_list = [name for name in seq_list if name == args.seq_name]
    if args.limit_seq > 0:
        seq_list = seq_list[:args.limit_seq]
    if not seq_list:
        raise SystemExit("No sequences to process.")

    mot = VMambaMOT(args)
    if args.show_params:
        print("\n=== VMamba Detector Model ===")
        print_model_params(mot.detector.model, "VMamba Detector")
        print("\n=== ResNet-50 ReID Model ===")
        print_model_params(mot.reid.model, "ResNet-50 ReID")

    print(f"\nProcessing {len(seq_list)} sequence(s)")
    print(f"Images: {image_split_dir}")
    print(f"GT root: {gt_split_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Detector: VMamba (Mask R-CNN)")
    print(f"ReID: ResNet-50")

    for seq_idx, seq_name in enumerate(seq_list, start=1):
        print("\n" + "=" * 60)
        print(f"[{seq_idx}/{len(seq_list)}] Processing {seq_name}")
        print("=" * 60)

        image_dir = os.path.join(image_split_dir, seq_name)
        image_files = list_image_files(image_dir)
        if args.limit_frames > 0:
            image_files = image_files[:args.limit_frames]
        if not image_files:
            print(f"Warning: no images in {image_dir}")
            continue

        output_seq_dir = os.path.join(args.output_dir, args.split, seq_name)
        vis_dir = os.path.join(args.output_dir, "vis", args.split, seq_name)
        os.makedirs(output_seq_dir, exist_ok=True)
        output_txt = os.path.join(output_seq_dir, "gt.txt")

        mot.reset()
        with open(output_txt, "w", encoding="utf-8") as out_f:
            for fallback_idx, image_path in enumerate(image_files, start=1):
                frame_id = frame_id_from_path(image_path, fallback_idx)
                detections = mot.track_frame(image_path, frame_id)
                write_mot_lines(out_f, frame_id, detections)
                out_f.flush()

                if args.vis:
                    vis_path = os.path.join(vis_dir, f"{frame_id:05d}.jpg")
                    visualize_frame(image_path, detections, vis_path)

                if not args.show_stats:
                    print(f"Frame {frame_id}: det={len(detections)}")

        print(f"Results saved to: {output_txt}")
        if args.vis:
            print(f"Visualization saved to: {vis_dir}")

    if args.show_stats or args.show_stats_power:
        mot.print_stats()

    print("\n" + "=" * 60)
    print("Inference complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

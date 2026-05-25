#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MambaVision detection + ReID multi-object tracking.

Pipeline:
1. MambaVision (via MMDetection) detects objects.
2. ReID associator tracks boxes across frames with IoU + appearance similarity.
3. Writes MOT-format results.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("HF_MODULES_CACHE", "/home/lvxuan/.cache/huggingface/modules")

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

DEFAULT_IMAGE_DIR = "/home/lvxuan/data/dwtest/images/train"
DEFAULT_GT_ROOT = "/home/lvxuan/data/dwtest/gt"
DEFAULT_OUTPUT_DIR = "/home/lvxuan/llmtest/outputs/mambavision_mot"
DEFAULT_MAMBAVISION_CONFIG = "/home/lvxuan/little/MambaVision-main/object_detection/configs/mamba_vision/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.py"
DEFAULT_MAMBAVISION_CHECKPOINT = "/home/lvxuan/little/MambaVision-main/object_detection/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth"
DEFAULT_MAMBAVISION_SOURCE = "/home/lvxuan/little/MambaVision-main"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

PERSON_CLASS_ID = 0

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128),
    (128, 255, 0), (0, 255, 128), (128, 0, 255), (0, 128, 255),
    (255, 128, 128), (128, 255, 128),
]


def numeric_sort_key(path: str) -> Tuple[int, Any]:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


def frame_id_from_path(path: str, fallback: int = 0) -> int:
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


def write_mot_lines(handle: Any, frame_id: int, detections: Sequence[Dict[str, Any]]) -> None:
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
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_top = max(0, y1 - text_h - baseline)
        cv2.rectangle(image, (x1, label_top), (x1 + text_w, y1), color, -1)
        cv2.putText(image, label, (x1, max(text_h, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


class ReIDAssociator:
    """Simple ReID-based associator."""

    def __init__(
        self,
        iou_weight: float = 0.35,
        feature_weight: float = 0.45,
        distance_weight: float = 0.20,
        iou_threshold: float = 0.10,
        feature_threshold: float = 0.45,
        dist_threshold: float = 180.0,
        match_threshold: float = 0.30,
        max_age: int = 20,
        feature_momentum: float = 0.80,
        id_start: int = 1,
        strong_iou_threshold: float = 0.25,
        appearance_only_threshold: float = 0.75,
        max_center_jump_ratio: float = 0.80,
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

        self.tracks: List[Dict[str, Any]] = []
        self.next_id = id_start

    def reset(self) -> None:
        self.tracks = []
        self.next_id = self.id_start

    def _associate_track(self, detection: Dict[str, Any], feature: torch.Tensor) -> Optional[int]:
        if not self.tracks:
            return None

        det_box = detection["bbox"]
        best_score = 0.0
        best_track_idx = None

        for track_idx, track in enumerate(self.tracks):
            if track["age"] > self.max_age:
                continue

            track_box = track["box"]
            iou = calculate_iou(det_box, track_box)
            if iou < self.iou_threshold:
                continue

            track_feature = track["feature"]
            feature_sim = torch.dot(feature, track_feature).item()

            det_center = ((det_box[0] + det_box[2]) / 2, (det_box[1] + det_box[3]) / 2)
            track_center = ((track_box[0] + track_box[2]) / 2, (track_box[1] + track_box[3]) / 2)
            dist = ((det_center[0] - track_center[0]) ** 2 + (det_center[1] - track_center[1]) ** 2) ** 0.5
            dist_score = 1.0 / (1.0 + dist / 100.0)

            score = (
                self.iou_weight * iou +
                self.feature_weight * feature_sim +
                self.distance_weight * dist_score
            )

            if score > best_score and score >= self.match_threshold:
                best_score = score
                best_track_idx = track_idx

        return best_track_idx

    def associate(
        self,
        detections: Sequence[Dict[str, Any]],
        features: torch.Tensor,
        frame_id: int,
    ) -> List[Dict[str, Any]]:
        tracked_detections: List[Dict[str, Any]] = []
        det_used = [False] * len(detections)

        # Associate existing tracks
        for track_idx, track in enumerate(self.tracks):
            if track["age"] > self.max_age:
                continue

            best_det_idx = None
            best_score = 0.0

            for det_idx, (det, feat) in enumerate(zip(detections, features)):
                if det_used[det_idx]:
                    continue

                iou = calculate_iou(track["box"], det["bbox"])
                feature_sim = torch.dot(feat, track["feature"]).item()

                score = self.iou_weight * iou + self.feature_weight * feature_sim
                if score > best_score and score >= self.match_threshold:
                    best_score = score
                    best_det_idx = det_idx

            if best_det_idx is not None:
                det = detections[best_det_idx]
                feat = features[best_det_idx]
                det_used[best_det_idx] = True

                # Update track
                new_box = det["bbox"]
                new_feature = (
                    self.feature_momentum * track["feature"] +
                    (1.0 - self.feature_momentum) * feat
                )
                new_feature = torch.nn.functional.normalize(new_feature, p=2, dim=-1)

                self.tracks[track_idx]["box"] = new_box
                self.tracks[track_idx]["feature"] = new_feature
                self.tracks[track_idx]["age"] = 0
                self.tracks[track_idx]["frame"] = frame_id

                tracked_det = dict(det)
                tracked_det["id"] = track["id"]
                tracked_detections.append(tracked_det)

        # Create new tracks for unassociated detections
        for det_idx, (det, feat) in enumerate(zip(detections, features)):
            if det_used[det_idx]:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks.append({
                "id": track_id,
                "box": det["bbox"],
                "feature": feat,
                "age": 0,
                "frame": frame_id,
                "velocity": [0.0, 0.0, 0.0, 0.0],
            })

            tracked_det = dict(det)
            tracked_det["id"] = track_id
            tracked_detections.append(tracked_det)

        # Age tracks
        for track in self.tracks:
            if track["frame"] != frame_id:
                track["age"] += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]

        return tracked_detections


class MambaVisionDetector:
    """Object detector using MambaVision with MMDetection."""

    def __init__(
        self,
        config_path: str = DEFAULT_MAMBAVISION_CONFIG,
        checkpoint_path: str = DEFAULT_MAMBAVISION_CHECKPOINT,
        mambavision_source_dir: str = DEFAULT_MAMBAVISION_SOURCE,
        box_threshold: float = 0.5,
        nms_iou: float = 0.5,
        max_det: int = 20,
        person_only: bool = True,
        device: str = "cuda",
    ):
        self.box_threshold = box_threshold
        self.nms_iou = nms_iou
        self.max_det = max_det
        self.person_only = person_only
        self.device = device
        self.config_path = config_path

        print("Loading MambaVision detector...")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")

        # Add MambaVision object_detection/tools to path
        obj_det_tools = os.path.join(mambavision_source_dir, "object_detection", "tools")
        if obj_det_tools not in sys.path:
            sys.path.insert(0, obj_det_tools)

        # Import mamba_vision module from object_detection/tools to register MM_mamba_vision
        import mamba_vision  # This registers MM_mamba_vision

        from mmengine.config import Config
        from mmdet.models import build_detector

        cfg = Config.fromfile(config_path)
        cfg.model.pretrained = None
        # Remove data_preprocessor for mmdet 2.x compatibility
        if "data_preprocessor" in cfg.model:
            del cfg.model["data_preprocessor"]
        # Set backbone pretrained to None to avoid loading default checkpoint
        if "backbone" in cfg.model and hasattr(cfg.model.backbone, "pretrained"):
            cfg.model.backbone.pretrained = None
        self.cfg = cfg
        self.model = build_detector(cfg.model, test_cfg=None)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        print(f"MambaVision detector loaded on {device}")

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in image using MMDetection."""
        try:
            return self._detect_mmdet(image_bgr)
        except Exception as e:
            print(f"MMDetection failed: {e}, falling back to simple detector")
            return self._detect_simple(image_bgr)

    def _detect_mmdet(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect using MMDetection."""
        import mmcv
        from mmdet.apis import inference_detector

        # Reload model with cfg for inference
        current_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        from mmdet.models import build_detector
        model_for_inference = build_detector(self.cfg.model, test_cfg=None)
        model_for_inference.load_state_dict(current_state)
        model_for_inference.eval().to(self.device)
        model_for_inference.cfg = self.cfg

        with torch.inference_mode():
            result = inference_detector(model_for_inference, image_bgr)

        detections = []
        bboxes = result[0]

        for bbox in bboxes:
            x1, y1, x2, y2, score = bbox[:5]
            label = int(bbox[5]) if len(bbox) > 5 else 0

            if self.person_only and label != PERSON_CLASS_ID:
                continue
            if score < self.box_threshold:
                continue

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
                "label": label,
            })

        if len(detections) > 0 and self.nms_iou > 0:
            boxes = np.array([d["bbox"] for d in detections])
            scores = np.array([d["score"] for d in detections])

            boxes_tensor = torch.from_numpy(boxes)
            scores_tensor = torch.from_numpy(scores)
            keep = nms(boxes_tensor, scores_tensor, self.nms_iou)

            if len(keep) < len(detections):
                detections = [detections[i] for i in keep.cpu().numpy()]

        if len(detections) > self.max_det:
            detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:self.max_det]

        return detections

    def _detect_simple(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Simple background subtraction detector as fallback."""
        if not hasattr(self, 'bg_subtractor'):
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16.0, detectShadows=False
            )
            self._bg_initialized = False

        if not self._bg_initialized:
            for _ in range(10):
                self.bg_subtractor.apply(image_bgr)
            self._bg_initialized = True

        fg_mask = self.bg_subtractor.apply(image_bgr)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        height, width = image_bgr.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
            detections.append({
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "score": 0.8,
                "label": 0,
            })

        # Apply NMS
        if len(detections) > 0 and self.nms_iou > 0:
            boxes = np.array([d["bbox"] for d in detections])
            scores = np.array([d["score"] for d in detections])
            boxes_tensor = torch.from_numpy(boxes)
            scores_tensor = torch.from_numpy(scores)
            keep = nms(boxes_tensor, scores_tensor, self.nms_iou)
            if len(keep) < len(detections):
                detections = [detections[i] for i in keep.cpu().numpy()]

        if len(detections) > self.max_det:
            detections = sorted(detections, key=lambda x: x["score"], reverse=True)[:self.max_det]

        return detections


class MambaVisionReIDExtractor:
    """ReID feature extractor using MambaVision backbone."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_MAMBAVISION_CHECKPOINT,
        mambavision_source_dir: str = DEFAULT_MAMBAVISION_SOURCE,
        batch_size: int = 4,
        crop_pad: float = 0.08,
        img_size: int = 224,
        dtype_name: str = "fp16",
    ):
        self.batch_size = max(1, batch_size)
        self.crop_pad = crop_pad
        self.img_size = img_size
        self.checkpoint_path = checkpoint_path
        self.mambavision_source_dir = mambavision_source_dir
        self.dtype = self._resolve_dtype(dtype_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.dtype = torch.float32

        print("Loading MambaVision for ReID...")
        print(f"Checkpoint: {checkpoint_path}")

        if mambavision_source_dir and os.path.isdir(mambavision_source_dir):
            if mambavision_source_dir not in sys.path:
                sys.path.insert(0, mambavision_source_dir)

        self.model = self._load_model()
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.transform = self._build_transform()
        print(f"MambaVision ReID loaded on {self.device}, dtype={self.dtype}")

    def _load_model(self) -> torch.nn.Module:
        from mambavision.models.mamba_vision import MambaVision

        model = MambaVision(
            dim=80,
            in_dim=32,
            depths=(1, 3, 8, 4),
            window_size=(8, 8, 14, 7),
            mlp_ratio=4.0,
            num_heads=(2, 4, 8, 16),
            drop_path_rate=0.1,
            in_chans=3,
            num_classes=0,
        )

        if os.path.isfile(self.checkpoint_path):
            print(f"Loading checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            backbone_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    backbone_dict[key[9:]] = value
                elif not any(key.startswith(p) for p in ["roi_head", "neck", "bbox_head", "rpn"]):
                    backbone_dict[key] = value

            result = model.load_state_dict(backbone_dict, strict=False)
            if isinstance(result, tuple) and len(result) == 2:
                missing, unexpected = result
                missing_keys = len(missing) if isinstance(missing, list) else 0
                unexpected_keys = len(unexpected) if isinstance(unexpected, list) else 0
            else:
                missing_keys, unexpected_keys = 0, 0
            print(f"Loaded backbone - missing: {missing_keys}, unexpected: {unexpected_keys}")

        return model

    def _build_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp32":
            return torch.float32
        if dtype_name == "auto":
            return torch.float16 if torch.cuda.is_available() else torch.float32
        return torch.float16

    def _crop_image(self, image_bgr: np.ndarray, box: Sequence[float]) -> Image.Image:
        height, width = image_bgr.shape[:2]
        crop_box = expand_box(box, width, height, self.crop_pad)
        if not crop_box:
            return None
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        patch = image_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return Image.fromarray(patch_rgb)

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
            return torch.zeros((len(detections), 640), dtype=torch.float32)

        features_by_det: List[torch.Tensor] = [None] * len(detections)

        with torch.inference_mode():
            for start in range(0, len(patch_images), self.batch_size):
                end = min(start + self.batch_size, len(patch_images))
                batch_patches = patch_images[start:end]

                tensors = [self.transform(p) for p in batch_patches]
                batch_tensor = torch.stack(tensors).to(device=self.device, dtype=self.dtype)

                if hasattr(self.model, "forward_features"):
                    features = self.model.forward_features(batch_tensor)
                else:
                    features = self.model(batch_tensor)

                if isinstance(features, (tuple, list)):
                    features = next(item for item in features if isinstance(item, torch.Tensor))
                if features.ndim > 2:
                    features = features.mean(dim=1)

                features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)

                for local_idx, det_idx in enumerate(valid_indices[start:end]):
                    features_by_det[det_idx] = features[local_idx].cpu()

        feature_dim = next((item for item in features_by_det if item is not None), torch.zeros(640)).numel()
        zero = torch.zeros(feature_dim, dtype=torch.float32)
        return torch.stack([item if item is not None else zero for item in features_by_det])


class MambaVisionMOT:
    """Multi-object tracker using MambaVision detection + ReID."""

    def __init__(self, args: argparse.Namespace):
        self.detector = MambaVisionDetector(
            config_path=args.mambavision_config,
            checkpoint_path=args.mambavision_checkpoint,
            mambavision_source_dir=args.mambavision_source_dir,
            box_threshold=args.box_threshold,
            nms_iou=args.det_nms_iou,
            max_det=args.max_det,
            person_only=args.person_only,
            device="cuda",
        )
        self.reid = MambaVisionReIDExtractor(
            checkpoint_path=args.mambavision_checkpoint,
            mambavision_source_dir=args.mambavision_source_dir,
            batch_size=args.reid_batch_size,
            crop_pad=args.reid_crop_pad,
            img_size=args.reid_img_size,
            dtype_name=args.dtype,
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
        self.fps_list: List[float] = []
        self.peak_memory_gb = 0.0

    def reset(self) -> None:
        self.associator.reset()
        self.fps_list = []
        self.peak_memory_gb = 0.0

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

        return tracked

    def print_stats(self) -> None:
        if not self.fps_list:
            return
        print("\n" + "=" * 60)
        print("Performance Statistics")
        print("=" * 60)
        print(f"Total frames: {len(self.fps_list)}")
        print(f"Peak GPU memory: {self.peak_memory_gb:.2f} GB")
        print(f"Average FPS: {np.mean(self.fps_list):.2f}")
        print(f"Median FPS: {np.median(self.fps_list):.2f}")
        print(f"Min FPS: {min(self.fps_list):.2f}")
        print(f"Max FPS: {max(self.fps_list):.2f}")
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MambaVision MOT")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--limit_seq", type=int, default=0)
    parser.add_argument("--limit_frames", type=int, default=0)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--show_stats", action="store_true")

    parser.add_argument("--mambavision_config", type=str, default=DEFAULT_MAMBAVISION_CONFIG)
    parser.add_argument("--mambavision_checkpoint", type=str, default=DEFAULT_MAMBAVISION_CHECKPOINT)
    parser.add_argument("--mambavision_source_dir", type=str, default=DEFAULT_MAMBAVISION_SOURCE)
    parser.add_argument("--person_only", action="store_true", default=True)
    parser.add_argument("--box_threshold", type=float, default=0.5)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    parser.add_argument("--det_nms_iou", type=float, default=0.5)
    parser.add_argument("--track_nms_iou", type=float, default=0.55)
    parser.add_argument("--max_det", type=int, default=20)

    parser.add_argument("--reid_batch_size", type=int, default=4)
    parser.add_argument("--reid_crop_pad", type=float, default=0.08)
    parser.add_argument("--reid_img_size", type=int, default=224)
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="fp16")

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

    if not os.path.isfile(args.mambavision_checkpoint):
        raise SystemExit(f"MambaVision checkpoint not found: {args.mambavision_checkpoint}")
    if not os.path.isfile(args.mambavision_config):
        raise SystemExit(f"MambaVision config not found: {args.mambavision_config}")

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

    mot = MambaVisionMOT(args)

    print(f"\nProcessing {len(seq_list)} sequence(s)")
    print(f"Images: {image_split_dir}")
    print(f"GT root: {gt_split_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Detector: MambaVision Tiny (MMDetection)")
    print(f"ReID: MambaVision Tiny")
    print(f"Checkpoint: {args.mambavision_checkpoint}")
    print(f"Box threshold: {args.box_threshold}")

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

    if args.show_stats:
        mot.print_stats()

    print("\n" + "=" * 60)
    print("Inference complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()

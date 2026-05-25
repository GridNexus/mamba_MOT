#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vim (Vision Mamba) based multi-object tracking for construction workers.

Pipeline:
1. Detect construction workers using a detector (MMYOLO/ GroundingDINO)
2. Extract visual appearance features using Vim for each detected box
3. Associate boxes across frames using geometry + Vim feature similarity
4. Write MOT-format results
"""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path

os.environ.setdefault("HF_MODULES_CACHE", "/data1/pengsihaoran/.cache/huggingface/modules")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor

# Import Vim model
import sys
sys.path.insert(0, "/data1/pengsihaoran/mamba/Vim-main/vim")
sys.path.insert(0, "/data1/pengsihaoran/mamba/Vim-main/mamba-1p1p1")

from models_mamba import VisionMamba


DEFAULT_IMAGE_DIR = "/data1/pengsihaoran/data/dwtest/images"
DEFAULT_GT_ROOT = "/data1/pengsihaoran/data/dwtest/gt"
DEFAULT_OUTPUT_DIR = "/data1/pengsihaoran/mamba/outputs/vim_mot"
DEFAULT_VIM_MODEL = "/data1/pengsihaoran/mamba/Vim-main/ckp/vim_b_midclstok_81p9acc.pth"
DEFAULT_DETECTOR_MODEL = "/data1/pengsihaoran/LLM/IDEA-Research/grounding-dino-tiny"
DEFAULT_PROMPT = "person . construction worker . worker ."

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


class HFGroundingDINODetector:
    """GroundingDINO detector for finding construction workers."""

    def __init__(
        self,
        model_path: str,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
        nms_iou: float,
        max_det: int,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.nms_iou = nms_iou
        self.max_det = max_det

        print("Loading HuggingFace GroundingDINO...")
        print(f"Model: {model_path}")

        from transformers import AutoModelForZeroShotObjectDetection

        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        print(f"GroundingDINO loaded on {self.device}.")

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        height, width = image_bgr.shape[:2]
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        text = self.prompt.strip()
        if not text.endswith("."):
            text += "."

        inputs = self.processor(images=image_pil, text=text, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[(height, width)],
        )[0]

        boxes = results.get("boxes", torch.empty((0, 4))).detach().cpu()
        scores = results.get("scores", torch.empty((0,))).detach().cpu()
        if boxes.numel() == 0:
            return []

        keep = nms(boxes, scores, self.nms_iou)
        detections = []
        for idx in keep:
            box = clip_box(boxes[idx].tolist(), width, height)
            if not box:
                continue
            detections.append({
                "bbox": box,
                "score": float(scores[idx].item()),
            })

        detections.sort(key=lambda item: item["score"], reverse=True)
        if self.max_det > 0:
            detections = detections[:self.max_det]
        return detections


class VimReID:
    """Vim (Vision Mamba) based appearance feature extractor for ReID."""

    def __init__(
        self,
        model_path: str,
        batch_size: int,
        crop_pad: float,
        dtype_name: str,
        img_size: int,
        reid_min_pixels: int,
        reid_max_pixels: int,
    ):
        self.batch_size = max(1, batch_size)
        self.crop_pad = crop_pad
        self.img_size = img_size
        self.reid_min_pixels = max(0, reid_min_pixels)
        self.reid_max_pixels = max(0, reid_max_pixels)
        self.dtype = self._resolve_dtype(dtype_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading Vim model for ReID...")
        print(f"Model: {model_path}")

        self.model = self._load_vim_model(model_path)
        self.model.to(self.device).eval()
        self.model_patch_size = self._get_patch_size()

        print(f"Vim ReID loaded. device: {self.device}, img_size: {self.img_size}")

    def _resolve_dtype(self, dtype_name: str) -> torch.dtype:
        if dtype_name == "bf16":
            return torch.bfloat16
        if dtype_name == "fp32":
            return torch.float32
        return torch.float16

    def _load_vim_model(self, model_path: str) -> VisionMamba:
        """Load Vim model from checkpoint."""
        model = VisionMamba(
            patch_size=16,
            embed_dim=768,
            d_state=16,
            depth=24,
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            final_pool_type='mean',
            if_abs_pos_embed=True,
            if_rope=False,
            if_rope_residual=False,
            bimamba_type="v2",
            if_cls_token=True,
            if_divide_out=True,
            use_middle_cls_token=True,
            img_size=self.img_size,
            num_classes=0,  # No classification head, return features
        )

        checkpoint = torch.load(model_path, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        # Remove classification head weights if present
        for k in list(checkpoint.keys()):
            if "head" in k:
                del checkpoint[k]

        model.load_state_dict(checkpoint, strict=False)
        return model

    def _get_patch_size(self) -> int:
        return self.model.patch_embed.patch_size[0]

    def _crop_image(self, image_bgr: np.ndarray, box: Sequence[float]) -> Optional[Image.Image]:
        height, width = image_bgr.shape[:2]
        crop_box = expand_box(box, width, height, self.crop_pad)
        if not crop_box:
            return None
        x1, y1, x2, y2 = crop_box
        patch = image_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return Image.fromarray(patch_rgb)

    def _resize_image(self, image: Image.Image, size: int) -> Image.Image:
        """Resize image to model input size."""
        return image.resize((size, size), Image.BICUBIC)

    def _normalize_image(self, image: Image.Image) -> torch.Tensor:
        """Normalize image for Vim model."""
        img_array = np.array(image).astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        return torch.from_numpy(img_array)

    def _visual_batch_features(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Extract features from a batch of images."""
        resized_images = [self._resize_image(img, self.img_size) for img in images]
        image_tensors = [self._normalize_image(img) for img in resized_images]
        batch_tensor = torch.stack(image_tensors).to(self.device)

        # Convert to float32 for Conv2d layers compatibility
        batch_tensor = batch_tensor.to(torch.float32)

        with torch.inference_mode():
            features = self.model.forward_features(batch_tensor)

        # Normalize features
        if features.ndim > 2:
            features = features.mean(dim=1)  # Pool spatial dimensions if needed

        features = torch.nn.functional.normalize(features.float(), p=2, dim=-1)
        return features.cpu()

    def extract_batch_features(
        self,
        image_bgr: np.ndarray,
        detections: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """Extract ReID features for all detections."""
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
                features = self._visual_batch_features(patch_images[start:end])
                for local_idx, det_idx in enumerate(valid_indices[start:end]):
                    features_by_det[det_idx] = features[local_idx]

        feature_dim = next(item for item in features_by_det if item is not None).numel()
        zero = torch.zeros(feature_dim, dtype=torch.float32)
        return torch.stack([item if item is not None else zero for item in features_by_det])


class ReIDAssociator:
    """Multi-object tracker with geometry + appearance matching."""

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

        # Stage 2: use geometry + Vim appearance for gaps, jitter, and occlusion
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

        # Assign new IDs for unmatched detections
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


class VimMOT:
    """Main MOT pipeline with Vim feature extraction."""

    def __init__(self, args: argparse.Namespace):
        self.detector = HFGroundingDINODetector(
            model_path=args.detector_model,
            prompt=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            nms_iou=args.det_nms_iou,
            max_det=args.max_det,
        )
        self.reid = VimReID(
            model_path=args.vim_model,
            batch_size=args.reid_batch_size,
            crop_pad=args.reid_crop_pad,
            dtype_name=args.dtype,
            img_size=args.img_size,
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
    """Write MOT format output."""
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
    """Draw tracking results on image."""
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
    parser = argparse.ArgumentParser(description="Vim-based Multi-Object Tracking")
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


    parser.add_argument("--detector_model", type=str, default=DEFAULT_DETECTOR_MODEL)
    parser.add_argument("--vim_model", type=str, default=DEFAULT_VIM_MODEL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--box_threshold", type=float, default=0.25)
    parser.add_argument("--text_threshold", type=float, default=0.20)
    parser.add_argument("--det_nms_iou", type=float, default=0.60)
    parser.add_argument("--track_nms_iou", type=float, default=0.55)
    parser.add_argument("--max_det", type=int, default=20)

    parser.add_argument("--reid_batch_size", type=int, default=4)
    parser.add_argument("--reid_crop_pad", type=float, default=0.08)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--reid_min_pixels", type=int, default=3136)
    parser.add_argument("--reid_max_pixels", type=int, default=200704)
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

    # Check model paths
    if not os.path.isfile(args.vim_model):
        raise SystemExit(f"Vim model not found: {args.vim_model}")

    if not os.path.isdir(args.detector_model):
        raise SystemExit(f"Detector model not found: {args.detector_model}")

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

    mot = VimMOT(args)

    
    if args.show_params:
        print("\n=== GroundingDINO Detector Model ===")
        print_model_params(mot.detector.model, "GroundingDINO Detector")
        print("\n=== Vim ReID Model ===")
        print_model_params(mot.reid.model, "Vim ReID")

    print(f"\nProcessing {len(seq_list)} sequence(s)")
    print(f"Images: {image_split_dir}")
    print(f"GT root: {gt_split_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Detector: {args.detector_model}")
    print(f"ReID: {args.vim_model}")
    print(f"Prompt: {args.prompt!r}")

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

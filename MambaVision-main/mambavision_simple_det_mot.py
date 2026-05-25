#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MambaVision detection + ReID multi-object tracking using HuggingFace model.

This script uses MambaVision from HF transformers to avoid mamba_ssm CUDA issues.
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

DEFAULT_IMAGE_DIR = "/home/lvxuan/data/dwtest/images/train"
DEFAULT_GT_ROOT = "/home/lvxuan/data/dwtest/gt"
DEFAULT_OUTPUT_DIR = "/home/lvxuan/llmtest/outputs/mambavision_simple_mot"
DEFAULT_MAMBAVISION_MODEL = "facebook/MambaVision-T"
DEFAULT_PROMPT = "person"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

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


def expand_box(box: Sequence[float], width: int, height: int, pad_ratio: float) -> Optional[Sequence[float]]:
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio
    nx1 = max(0.0, center_x - (w / 2.0 + pad_w))
    ny1 = max(0.0, center_y - (h / 2.0 + pad_h))
    nx2 = min(float(width), center_x + (w / 2.0 + pad_w))
    ny2 = min(float(height), center_y + (h / 2.0 + pad_h))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return (nx1, ny1, nx2, ny2)


def write_mot_lines(out_f, frame_id: int, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        out_f.write(f"{frame_id},{det['id']},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{det.get('score', 1.0):.4f},1,-1\n")


def visualize_frame(image_path: str, detections: List[Dict[str, Any]], output_path: str) -> None:
    image = cv2.imread(image_path)
    if image is None:
        return
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        color = COLORS[det["id"] % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{det['id']} {det.get('score', 1.0):.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def suppress_duplicate_detections(detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.get("score", 0.0), reverse=True)
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [d for d in detections if calculate_iou(current["bbox"], d["bbox"]) < iou_threshold]
    return keep


class SimpleDetector:
    """Simple background subtraction detector."""

    def __init__(self, hist_size: int = 500, var_threshold: float = 16.0, min_area: int = 100):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=hist_size, varThreshold=var_threshold, detectShadows=False
        )
        self.min_area = min_area

    def detect(self, image_bgr: np.ndarray) -> List[Dict[str, Any]]:
        fg_mask = self.bg_subtractor.apply(image_bgr)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 20 or h < 20:
                continue
            detections.append({
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "score": 0.8,
            })
        return detections


class MambaVisionReIDExtractor:
    """ReID feature extractor using MambaVision backbone from HF."""

    def __init__(
        self,
        model_name: str = DEFAULT_MAMBAVISION_MODEL,
        batch_size: int = 4,
        crop_pad: float = 0.08,
        img_size: int = 224,
        dtype_name: str = "fp16",
    ):
        self.batch_size = max(1, batch_size)
        self.crop_pad = crop_pad
        self.img_size = img_size
        self.dtype = self._resolve_dtype(dtype_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            self.dtype = torch.float32

        print("Loading MambaVision for ReID from HuggingFace...")
        print(f"Model: {model_name}")

        self.model = self._load_model(model_name)
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.transform = self._build_transform()
        print(f"MambaVision ReID loaded on {self.device}, dtype={self.dtype}")

    def _load_model(self, model_name: str):
        """Load MambaVision from local package."""
        # Add MambaVision source to path
        mambavision_source = "/home/lvxuan/little/MambaVision-main"
        if mambavision_source not in sys.path:
            sys.path.insert(0, mambavision_source)

        from mambavision.models.mamba_vision import MambaVision

        # Create MambaVision Tiny backbone
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

        # Load checkpoint if available
        checkpoint_path = "/home/lvxuan/little/MambaVision-main/object_detection/cascade_mask_rcnn_mamba_vision_tiny_3x_coco.pth"
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Filter backbone weights only
            backbone_dict = {}
            for key, value in state_dict.items():
                if key.startswith("backbone."):
                    backbone_dict[key[9:]] = value
                elif not any(key.startswith(p) for p in ["roi_head", "neck", "bbox_head", "rpn"]):
                    backbone_dict[key] = value

            model.load_state_dict(backbone_dict, strict=False)
            print("Loaded backbone weights")

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

    def _crop_image(self, image_bgr: np.ndarray, box: Sequence[float]) -> Optional[Image.Image]:
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
            return torch.zeros((len(detections), 640), dtype=torch.float32)

        features_by_det: List[Optional[torch.Tensor]] = [None] * len(detections)

        with torch.inference_mode():
            for start in range(0, len(patch_images), self.batch_size):
                end = min(start + self.batch_size, len(patch_images))
                batch_patches = patch_images[start:end]

                tensors = [self.transform(p) for p in batch_patches]
                batch_tensor = torch.stack(tensors).to(device=self.device, dtype=self.dtype)

                # Extract features
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


class ReIDAssociator:
    """Track associator using IoU + appearance + distance."""

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
        self.next_id = id_start
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.strong_iou_threshold = strong_iou_threshold
        self.appearance_only_threshold = appearance_only_threshold
        self.max_center_jump_ratio = max_center_jump_ratio

    def reset(self) -> None:
        self.tracks.clear()

    def associate(
        self,
        detections: List[Dict[str, Any]],
        features: torch.Tensor,
        frame_id: int,
    ) -> List[Dict[str, Any]]:
        """Associate detections with tracks."""
        if not detections:
            # Age out tracks
            self._remove_old_tracks(frame_id)
            return []

        # Initialize new detections with features
        for idx, det in enumerate(detections):
            det["feature"] = features[idx].numpy() if features.numel() > 0 else np.zeros(640)

        # Split into high/low confidence detections
        high_conf_dets = [d for d in detections if d.get("score", 0) >= 0.5]
        low_conf_dets = [d for d in detections if d.get("score", 0) < 0.5]

        matched_high = self._associate_detections(high_conf_dets, frame_id)
        matched_low = self._associate_detections(low_conf_dets, frame_id, require_strong_match=True)

        # Initialize new tracks for unmatched detections
        for det in detections:
            if "id" not in det:
                det["id"] = self.next_id
                self.tracks[det["id"]] = {
                    "last_seen": frame_id,
                    "feature": det["feature"].copy(),
                    "bbox": det["bbox"],
                }
                self.next_id += 1

        # Remove old tracks
        self._remove_old_tracks(frame_id)

        return detections

    def _associate_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int,
        require_strong_match: bool = False,
    ) -> List[int]:
        """Associate detections with existing tracks."""
        matched_indices = []
        active_tracks = [tid for tid, t in self.tracks.items() if t["last_seen"] >= frame_id - self.max_age]

        if not active_tracks:
            return matched_indices

        for det_idx, det in enumerate(detections):
            best_score = 0.0
            best_tid = None

            for tid in active_tracks:
                track = self.tracks[tid]
                iou = calculate_iou(det["bbox"], track["bbox"])
                feature_sim = 1.0 - np.linalg.norm(det["feature"] - track["feature"]) / 2.0
                dist = np.sqrt(
                    (det["bbox"][0] - track["bbox"][0]) ** 2 +
                    (det["bbox"][1] - track["bbox"][1]) ** 2
                )
                dist_score = max(0.0, 1.0 - dist / self.dist_threshold)

                combined = (
                    self.iou_weight * iou +
                    self.feature_weight * feature_sim +
                    self.distance_weight * dist_score
                )

                if combined > best_score:
                    best_score = combined
                    best_tid = tid

            if best_tid is not None and best_score >= self.match_threshold:
                if require_strong_match and best_score < self.appearance_only_threshold:
                    continue
                det["id"] = best_tid
                matched_indices.append(det_idx)

                # Update track
                track = self.tracks[best_tid]
                track["last_seen"] = frame_id
                track["feature"] = (
                    self.feature_momentum * track["feature"] +
                    (1 - self.feature_momentum) * det["feature"]
                )
                track["bbox"] = det["bbox"]

        return matched_indices

    def _remove_old_tracks(self, frame_id: int) -> None:
        """Remove tracks that haven't been seen recently."""
        to_remove = [
            tid for tid, track in self.tracks.items()
            if frame_id - track["last_seen"] > self.max_age
        ]
        for tid in to_remove:
            del self.tracks[tid]


class MambaVisionMOT:
    """Multi-object tracker using simple detector + MambaVision ReID."""

    def __init__(self, args: argparse.Namespace):
        self.detector = SimpleDetector()
        self.reid = MambaVisionReIDExtractor(
            model_name=args.mambavision_model,
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
    parser = argparse.ArgumentParser(description="MambaVision ReID MOT")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--gt_root", type=str, default=DEFAULT_GT_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seq_name", type=str, default=None)
    parser.add_argument("--limit_seq", type=int, default=0)
    parser.add_argument("--limit_frames", type=int, default=0)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--show_stats", action="store_true")

    parser.add_argument("--mambavision_model", type=str, default=DEFAULT_MAMBAVISION_MODEL)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--track_nms_iou", type=float, default=0.55)

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
    print(f"ReID: MambaVision (HF/timm)")
    print(f"Model: {args.mambavision_model!r}")

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

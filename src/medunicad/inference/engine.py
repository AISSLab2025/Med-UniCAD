"""Deployment-style single-request inference for the unified Swin + MoFX model.

The script accepts one image, modality, and one or more requested tasks. It
expands cross-task dependencies, performs the required model passes, and writes
all outputs under:

    <output_root>/<request_id>/

Example:
    python unified_medical_inference.py \
      --config inference_config.example.json \
      --image sample.png \
      --modality ultrasound \
      --tasks segmentation,birads,pathology,report \
      --request-id US_0001

Important:
- Model architecture fields must match the training run exactly.
- Class-name ordering must match the checkpoint exactly.
- This is inference code only; no GT mask, boxes, labels, or report are needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoTokenizer

from medunicad.modeling.freqmoe_dense import MoFXDense, SwinTransformer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

TASK_ALIASES = {
    "seg": "segmentation",
    "segment": "segmentation",
    "segmentation": "segmentation",
    "cls": "classification",
    "classify": "classification",
    "classification": "classification",
    "birads": "birads",
    "bi-rads": "birads",
    "bi_rads": "birads",
    "path": "pathology",
    "pathology": "pathology",
    "det": "detection",
    "detect": "detection",
    "detection": "detection",
    "severity": "severity",
    "mri_severity": "severity",
    "roi_severity": "severity",
    "rep": "report",
    "report": "report",
    "report_generation": "report",
    "all": "all",
}

MODALITY_ALIASES = {
    "xray": "xray",
    "x-ray": "xray",
    "x_ray": "xray",
    "cxr": "xray",
    "radiograph": "xray",
    "ct": "ct",
    "computed_tomography": "ct",
    "mri": "mri",
    "mr": "mri",
    "ultrasound": "ultrasound",
    "us": "ultrasound",
    "sonography": "ultrasound",
}


@dataclass
class InferenceConfig:
    # Files and runtime
    checkpoint_path: str
    dino_repo_path: str
    dino_cfg_path: str
    dino_pretrained_path: str
    llm_model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    medclip_path: Optional[str] = None
    output_root: str = "Inference Results"
    device: str = "cuda"
    target_size: Tuple[int, int] = (224, 224)
    use_amp: bool = True

    # Dataset/model metadata. Ordering must be identical to training.
    # Exact training-time modality IDs:
    # CT=0, MRI=1, Ultrasound=2, XRay=3. Config keys are normalized through
    # MODALITY_ALIASES, so both training names and CLI aliases are accepted.
    modality_map: Dict[str, int] = field(default_factory=lambda: {
        "CT": 0, "MRI": 1, "Ultrasound": 2, "XRay": 3
    })
    class_names: List[str] = field(default_factory=list)
    segmentation_class_names: List[str] = field(default_factory=lambda: ["foreground"])
    segmentation_classes_by_modality: Dict[str, List[str]] = field(default_factory=dict)
    classification_classes_by_modality: Dict[str, List[str]] = field(default_factory=dict)
    detection_class_names: List[str] = field(default_factory=lambda: [
        "L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1"
    ])
    severity_class_names: List[str] = field(default_factory=lambda: [
        "Normal", "Stenosis", "Severe Stenosis"
    ])
    severity_class_indices: List[int] = field(default_factory=list)
    birads_class_names: List[str] = field(default_factory=lambda: ["2", "3", "4", "5"])
    pathology_class_names: List[str] = field(default_factory=lambda: ["benign", "malignant"])

    # Thresholds
    segmentation_threshold: float = 0.5
    classification_threshold: float = 0.5
    detection_score_threshold: float = 0.3
    detection_nms_iou: float = 0.4
    detection_topk: int = 20
    birads_thresholds: Optional[List[float]] = None

    # Active model architecture. These values mirror the dense FreqXpert setup.
    use_moe: bool = True
    moe_type: str = "gradif"
    gradif_routing_mode: str = "dense_soft"
    gradif_expert_bank_type: str = "freqxpert"
    gradif_decomposition: str = "gaussian_fixed"
    gradif_active_stages: Tuple[str, ...] = ("c2", "c3", "c4_semantic")
    gradif_include_generalist: bool = False
    gradif_include_null_expert: bool = True
    gradif_top_k: int = 2
    gradif_temperature: float = 1.5
    gradif_attn_dim: int = 96
    gradif_num_heads: int = 4
    gradif_router_dim: int = 128
    gradif_modality_scale: float = 1.0
    gradif_router_strategy: str = "factorized_multigate"
    gradif_router_alpha_content_init: float = 0.50
    gradif_router_alpha_task_init: float = 0.25
    gradif_router_alpha_modality_init: float = 0.25
    gradif_router_task_specific_gates: bool = True
    gradif_router_modality_specific_gates: bool = True
    gradif_router_modality_content_adapters: bool = True
    gradif_router_branch_hidden: int = 96
    mofx_router_configuration: str = "full"
    gradif_router_task_residual_scale: float = 0.10
    gradif_router_modality_residual_scale: float = 0.10
    gradif_router_attention_pool: bool = True
    gradif_router_mi_min_group_size: int = 4
    gradif_router_mi_queue_size: int = 32
    gradif_router_mi_current_fraction: float = 0.50
    gradif_gamma_init: float = 0.015
    gradif_gamma_min: float = 0.0
    gradif_gamma_max: float = 0.12
    gradif_expert_scale_init: float = 0.35
    gradif_expert_scale_max: float = 1.0
    gradif_generic_mlp_bottleneck: int = 96
    gradif_use_granularity_prior: bool = False
    classification_gradif_fusion_init: float = 0.25
    classification_gradif_max_delta_ratio: float = 0.06
    segmentation_gradif_fusion_init: float = 0.85
    segmentation_gradif_max_delta_ratios: Tuple[float, float, float] = (0.05, 0.04, 0.05)
    shared_head_seed: int = 1337

    # Generation
    report_num_beams: int = 3
    report_max_new_tokens: int = 180
    report_do_sample: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "InferenceConfig":
        config_path = Path(path).expanduser().resolve(strict=True)
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Resolve all file-system fields relative to the JSON file, not the
        # current working directory.
        base_dir = config_path.parent
        for key in (
            "checkpoint_path",
            "dino_repo_path",
            "dino_cfg_path",
            "dino_pretrained_path",
            "medclip_path",
            "output_root",
        ):
            value = raw.get(key)
            if value in (None, ""):
                continue

            candidate = Path(str(value)).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate

            raw[key] = str(candidate.resolve(strict=False))

        for key in (
            "target_size", "gradif_active_stages",
            "segmentation_gradif_max_delta_ratios",
        ):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)


@dataclass
class InferenceRequest:
    image_path: str
    modality: str
    tasks: List[str]
    request_id: Optional[str] = None
    save_input_copy: bool = True


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def _safe_id(value: Optional[str]) -> str:
    if value:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
        if cleaned:
            return cleaned
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"REQ_{stamp}_{uuid.uuid4().hex[:8]}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, indent=2, ensure_ascii=False)


def _load_state_dict_flexible(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
) -> Dict[str, Any]:
    """
    Load the inference checkpoint using the same rich/legacy checkpoint
    distinction as the training resume logic.

    It additionally normalizes torch.compile prefixes. During training, the
    compiled Swin backbone may be saved as ``swin._orig_mod.*`` while the
    inference model expects ``swin.*``.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # Same distinction used by the original training resume function.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("  [Info] Detected Rich Checkpoint.")
        state_dict = checkpoint["model_state_dict"]
        checkpoint_format = "rich"
        checkpoint_epoch = checkpoint.get("epoch")
    elif isinstance(checkpoint, dict):
        print("  [Warning] Detected Legacy Checkpoint. Loading weights only.")
        state_dict = checkpoint
        checkpoint_format = "legacy"
        checkpoint_epoch = None
    else:
        raise TypeError(
            "Checkpoint must be a state dictionary or contain "
            "'model_state_dict'."
        )

    if not isinstance(state_dict, dict):
        raise TypeError("The selected checkpoint state dictionary is invalid.")

    model_state = model.state_dict()
    cleaned_state = {}

    for raw_key, value in state_dict.items():
        key = str(raw_key)

        # Remove common outer wrappers.
        while key.startswith("module."):
            key = key[len("module."):]

        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

        # torch.compile was applied only to the Swin member in training.
        key = key.replace("swin._orig_mod.", "swin.")

        cleaned_state[key] = value

    compatible_state = {}
    shape_mismatches = []
    checkpoint_only = []

    for key, value in cleaned_state.items():
        if key not in model_state:
            checkpoint_only.append(key)
            continue

        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": tuple(value.shape),
                    "model_shape": tuple(model_state[key].shape),
                }
            )
            continue

        compatible_state[key] = value

    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    print(
        f"[Checkpoint] loaded={len(compatible_state)}, "
        f"missing={len(missing)}, "
        f"checkpoint_only={len(checkpoint_only)}, "
        f"shape_mismatch={len(shape_mismatches)}"
    )

    if checkpoint_epoch is not None:
        print(f"  [Info] Checkpoint epoch: {checkpoint_epoch}")

    if missing:
        print("  Missing examples:", missing[:10])

    if checkpoint_only:
        print("  Checkpoint-only examples:", checkpoint_only[:10])

    if shape_mismatches:
        print("  Shape-mismatch examples:", shape_mismatches[:5])

    missing_swin = [key for key in missing if key.startswith("swin.")]
    loaded_swin = [key for key in compatible_state if key.startswith("swin.")]

    if missing_swin and not loaded_swin:
        raise RuntimeError(
            "No Swin backbone weights were loaded. The checkpoint/model "
            "architecture or torch.compile key normalization does not match."
        )

    del checkpoint
    del state_dict

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "checkpoint_format": checkpoint_format,
        "checkpoint_epoch": checkpoint_epoch,
        "loaded_keys": len(compatible_state),
        "missing_keys": missing,
        "checkpoint_only_keys": checkpoint_only,
        "shape_mismatches": shape_mismatches,
        "unexpected_keys": unexpected,
    }


def _pil_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = (tensor * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)


def _normalize_modality(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "_")
    if key not in MODALITY_ALIASES:
        raise ValueError(f"Unsupported modality: {value}")
    return MODALITY_ALIASES[key]


def _normalize_tasks(values: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for value in values:
        key = str(value).strip().lower().replace(" ", "_")
        if key not in TASK_ALIASES:
            raise ValueError(f"Unsupported task: {value}")
        out.add(TASK_ALIASES[key])
    return out


def _requested_all_for_modality(modality: str) -> Set[str]:
    if modality == "ultrasound":
        return {"segmentation", "birads", "pathology", "report"}
    if modality == "mri":
        return {"segmentation", "detection", "severity", "report"}
    if modality == "ct":
        return {"classification", "report"}
    return {"segmentation", "classification", "report"}


def resolve_dependencies(modality: str, requested: Set[str]) -> Tuple[Set[str], Set[str]]:
    """Return (requested_tasks, execution_tasks).

    Execution tasks may contain prerequisites that are not explicitly requested.
    """
    if "all" in requested:
        requested = _requested_all_for_modality(modality)

    allowed = {
        "xray": {"segmentation", "classification", "report"},
        "ct": {"classification", "report"},
        "mri": {"segmentation", "detection", "severity", "report"},
        "ultrasound": {"segmentation", "birads", "pathology", "report"},
    }[modality]
    invalid = requested - allowed
    if invalid:
        raise ValueError(f"Tasks {sorted(invalid)} are not supported for modality {modality}")

    execution = set(requested)

    # Cross-ROI / cross-task prerequisites.
    if modality == "ultrasound":
        if requested & {"classification", "birads", "pathology", "report"}:
            execution.add("segmentation")
        if "report" in requested:
            execution.update({"birads", "pathology"})
    elif modality == "xray":
        if requested & {"classification", "report"}:
            execution.add("segmentation")
        if "report" in requested:
            execution.add("classification")
    elif modality == "ct":
        if "report" in requested:
            execution.add("classification")
    elif modality == "mri":
        if requested & {"severity", "report"}:
            execution.add("detection")
        if "report" in requested:
            execution.add("severity")

    return requested, execution


# -----------------------------------------------------------------------------
# Visualization and measurements
# -----------------------------------------------------------------------------

def _binary_mask_from_logits(logits: torch.Tensor, threshold: float, output_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    if logits.ndim != 4:
        raise ValueError(f"Expected segmentation logits [B,C,H,W], got {tuple(logits.shape)}")
    probs = torch.sigmoid(logits.float())
    probs = F.interpolate(probs, size=output_hw, mode="bilinear", align_corners=False)
    if probs.shape[1] == 1:
        prob = probs[0, 0]
    else:
        prob = probs[0].amax(dim=0)
    return prob.detach().cpu().numpy(), (prob >= threshold).detach().cpu().numpy().astype(np.uint8)


def _multiclass_mask_from_logits(logits: torch.Tensor, output_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Legacy helper; modality-aware decoding below uses independent sigmoid channels."""
    logits = F.interpolate(logits.float(), size=output_hw, mode="bilinear", align_corners=False)
    probs = torch.softmax(logits, dim=1)[0]
    class_map = probs.argmax(dim=0)
    confidence = probs.max(dim=0).values
    return confidence.detach().cpu().numpy(), class_map.detach().cpu().numpy().astype(np.uint8)


def _resolve_named_indices(global_names: Sequence[str], selected_names: Sequence[str], field_name: str) -> List[int]:
    lookup = {str(name): idx for idx, name in enumerate(global_names)}
    missing = [str(name) for name in selected_names if str(name) not in lookup]
    if missing:
        raise ValueError(f"Unknown names in {field_name}: {missing}. Available: {list(global_names)}")
    return [lookup[str(name)] for name in selected_names]


def _modality_segmentation_from_logits(logits, global_names, selected_names, threshold, output_hw):
    indices = _resolve_named_indices(global_names, selected_names, "segmentation_classes_by_modality")
    if not indices:
        raise ValueError("No segmentation channels configured for this modality")
    resized = F.interpolate(logits.float(), size=output_hw, mode="bilinear", align_corners=False)
    selected = torch.sigmoid(resized[0, indices])
    probabilities = selected.detach().cpu().numpy()
    binary_masks = (selected >= float(threshold)).detach().cpu().numpy().astype(np.uint8)
    if len(indices) == 1:
        label_map = binary_masks[0]
        confidence = probabilities[0]
    else:
        max_prob, winner = selected.max(dim=0)
        label_map_t = winner.to(torch.uint8) + 1
        label_map_t[max_prob < float(threshold)] = 0
        label_map = label_map_t.detach().cpu().numpy().astype(np.uint8)
        confidence = max_prob.detach().cpu().numpy()
    return {"global_indices": indices, "class_names": list(selected_names),
            "probabilities": probabilities, "binary_masks": binary_masks,
            "label_map": label_map, "confidence": confidence}


def _segmentation_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    if mask.max() <= 1:
        fg = mask > 0
        color = np.zeros_like(overlay)
        color[..., 1] = 255
        overlay[fg] = 0.55 * overlay[fg] + 0.45 * color[fg]
    else:
        palette = np.array([
            [0, 0, 0], [31, 119, 180], [255, 127, 14], [44, 160, 44],
            [214, 39, 40], [148, 103, 189], [140, 86, 75], [227, 119, 194],
        ], dtype=np.float32)
        for cls_id in np.unique(mask):
            if cls_id == 0:
                continue
            fg = mask == cls_id
            color = palette[int(cls_id) % len(palette)]
            overlay[fg] = 0.55 * overlay[fg] + 0.45 * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def compute_ultrasound_morphology(mask: np.ndarray) -> Dict[str, Any]:
    contour = _largest_contour(mask)
    empty = {
        "max_diameter_px": 0.0,
        "perpendicular_diameter_px": 0.0,
        "bbox_width_px": 0.0,
        "bbox_height_px": 0.0,
        "aspect_ratio_h_over_w": 0.0,
        "area_px": 0.0,
        "perimeter_px": 0.0,
        "circularity": 0.0,
        "solidity": 0.0,
        "extent": 0.0,
        "eccentricity": 0.0,
    }
    if contour is None or cv2.contourArea(contour) <= 0:
        return {**empty, "valid": False}

    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))

    max_diameter = float(max(w, h))
    perpendicular = float(min(w, h))
    eccentricity = 0.0
    ellipse = None
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (_, _), (axis_a, axis_b), _ = ellipse
        major, minor = max(axis_a, axis_b), min(axis_a, axis_b)
        max_diameter, perpendicular = float(major), float(minor)
        eccentricity = float(math.sqrt(max(0.0, 1.0 - (minor * minor) / max(major * major, 1e-8))))

    return {
        "valid": True,
        "max_diameter_px": max_diameter,
        "perpendicular_diameter_px": perpendicular,
        "bbox_width_px": float(w),
        "bbox_height_px": float(h),
        "aspect_ratio_h_over_w": float(h / max(w, 1)),
        "area_px": area,
        "perimeter_px": perimeter,
        "circularity": float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-8)),
        "solidity": float(area / max(hull_area, 1e-8)),
        "extent": float(area / max(float(w * h), 1e-8)),
        "eccentricity": eccentricity,
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "ellipse": ellipse,
    }


def save_ultrasound_morphology_overlay(image_rgb: np.ndarray, mask: np.ndarray, measurements: Dict[str, Any], path: Path) -> None:
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    contour = _largest_contour(mask)
    if contour is not None:
        cv2.drawContours(canvas, [contour], -1, (0, 255, 0), 2)
        hull = cv2.convexHull(contour)
        cv2.drawContours(canvas, [hull], -1, (255, 200, 0), 1)
        x1, y1, x2, y2 = measurements.get("bbox", [0, 0, 0, 0])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)
        ellipse = measurements.get("ellipse")
        if ellipse is not None:
            cv2.ellipse(canvas, ellipse, (255, 0, 255), 2)
            (cx, cy), (a, b), angle = ellipse
            theta = math.radians(angle)
            if a < b:
                theta += math.pi / 2
                a, b = b, a
            major_vec = np.array([math.cos(theta), math.sin(theta)]) * a / 2
            minor_vec = np.array([-math.sin(theta), math.cos(theta)]) * b / 2
            center = np.array([cx, cy])
            p1, p2 = (center - major_vec).astype(int), (center + major_vec).astype(int)
            q1, q2 = (center - minor_vec).astype(int), (center + minor_vec).astype(int)
            cv2.arrowedLine(canvas, tuple(p1), tuple(p2), (0, 0, 255), 2, tipLength=0.05)
            cv2.arrowedLine(canvas, tuple(q1), tuple(q2), (255, 255, 0), 2, tipLength=0.05)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas)


def _draw_detection_overlay(
    image_rgb: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    class_names: Sequence[str],
    severity_by_detection: Optional[Mapping[int, Dict[str, Any]]] = None,
) -> np.ndarray:
    """Draw MRI level boxes, scores, and optional severity labels."""
    canvas = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(canvas)
    for idx in range(int(boxes.shape[0])):
        x1, y1, x2, y2 = [float(v) for v in boxes[idx].tolist()]
        label_id = int(labels[idx].item()) if labels.numel() else -1
        label = class_names[label_id] if 0 <= label_id < len(class_names) else str(label_id)
        score = float(scores[idx].item()) if scores.numel() else 0.0

        suffix = ""
        if severity_by_detection is not None:
            severity = severity_by_detection.get(idx)
            if severity is not None:
                suffix = f" | {severity.get('label', '')}"

        display_text = f"{label} {score:.2f}{suffix}"
        draw.rectangle((x1, y1, x2, y2), outline=(255, 70, 70), width=3)
        text_width = max(95, 7 * len(display_text))
        draw.rectangle(
            (x1, max(0, y1 - 18), min(canvas.width - 1, x1 + text_width), y1),
            fill=(0, 0, 0),
        )
        draw.text(
            (x1 + 2, max(0, y1 - 17)),
            display_text,
            fill=(255, 255, 255),
        )
    return np.asarray(canvas)


def _scale_boxes_xyxy(
    boxes: torch.Tensor,
    from_hw: Tuple[int, int],
    to_hw: Tuple[int, int],
) -> torch.Tensor:
    """Scale xyxy boxes between image coordinate systems."""
    if boxes.numel() == 0:
        return boxes.clone()

    from_h, from_w = int(from_hw[0]), int(from_hw[1])
    to_h, to_w = int(to_hw[0]), int(to_hw[1])
    if from_h <= 0 or from_w <= 0:
        raise ValueError(f"Invalid source image size: {from_hw}")

    scaled = boxes.clone().float()
    scaled[:, [0, 2]] *= float(to_w) / float(from_w)
    scaled[:, [1, 3]] *= float(to_h) / float(from_h)
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clamp(0, max(0, to_w - 1))
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clamp(0, max(0, to_h - 1))
    return scaled


def _severity_map_by_detection(
    severity_rows: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(row["detection_index"]): row
        for row in severity_rows
        if "detection_index" in row
    }


def _best_severity_per_level(
    severity_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return one severity prediction per anatomical level.

    When the detector emits duplicate boxes for the same level, the prediction
    attached to the highest-confidence detection is retained.
    """
    best: Dict[int, Dict[str, Any]] = {}
    for row in severity_rows:
        level_id = int(row["level_id"])
        previous = best.get(level_id)
        if previous is None or float(row.get("detection_score", 0.0)) > float(
            previous.get("detection_score", 0.0)
        ):
            best[level_id] = row
    return [best[key] for key in sorted(best)]


def _complete_mri_level_rows(
    detected_level_rows: Sequence[Dict[str, Any]],
    level_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Return a stable row for every configured MRI anatomical level."""
    detected = {
        int(row["level_id"]): row
        for row in detected_level_rows
    }
    complete: List[Dict[str, Any]] = []
    for level_id, level_name in enumerate(level_names):
        if level_id in detected:
            row = dict(detected[level_id])
            row["status"] = "detected"
            complete.append(row)
        else:
            complete.append({
                "level_id": level_id,
                "level": str(level_name),
                "status": "not_detected",
                "detection_index": None,
                "detection_score": None,
                "box_xyxy": None,
                "box_xyxy_model": None,
                "severity_id": None,
                "label": None,
                "probabilities": {},
                "crop_box_xyxy": None,
                "crop_box_xyxy_model": None,
                "crop_path": None,
            })
    return complete


def _save_gradcam(
    image_rgb: np.ndarray,
    feature: torch.Tensor,
    logits: torch.Tensor,
    target_index: int,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature.retain_grad()
    target = logits[0, target_index]
    target.backward(retain_graph=False)
    grad = feature.grad
    if grad is None:
        raise RuntimeError("Grad-CAM feature gradient is unavailable")

    weights = grad[0].float().mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * feature[0].float()).sum(dim=0))
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-8)
    cam = F.interpolate(cam[None, None], size=image_rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    cam_np = cam.detach().cpu().numpy()

    raw_path = out_dir / "gradcam.npy"
    np.save(raw_path, cam_np)
    heat = cv2.applyColorMap(np.uint8(cam_np * 255), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55 * image_rgb + 0.45 * heat_rgb, 0, 255).astype(np.uint8)
    heat_path = out_dir / "gradcam_heatmap.png"
    overlay_path = out_dir / "gradcam_overlay.png"
    _save_rgb(heat_path, heat_rgb)
    _save_rgb(overlay_path, overlay)
    return {"raw": str(raw_path), "heatmap": str(heat_path), "overlay": str(overlay_path)}


# -----------------------------------------------------------------------------
# Engine
# -----------------------------------------------------------------------------

class UnifiedInferenceEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.output_root = Path(
            self.config.output_root
        ).expanduser().resolve(strict=False)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.config.output_root = str(self.output_root)
        self._validate_label_configuration()
        self.device = self._resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = self._build_model().to(self.device)
        self.checkpoint_info = _load_state_dict_flexible(self.model, config.checkpoint_path)
        self.model.eval()
        self.model.use_gt_descriptors = False
        self.transform = transforms.Compose([
            transforms.Resize(config.target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        requested = str(requested).lower()
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[Warning] CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device(requested)

    def _build_model(self) -> torch.nn.Module:
        cfg = self.config
        swin = SwinTransformer(
            img_size=int(cfg.target_size[0]),
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
        )
        ultrasound_id = self._modality_id("ultrasound", allow_missing=True)
        model = MoFXDense(
            swin_model=swin,
            medclip_path=cfg.medclip_path,
            use_moe=cfg.use_moe,
            moe_type=cfg.moe_type,
            gradif_attn_dim=cfg.gradif_attn_dim,
            gradif_num_heads=cfg.gradif_num_heads,
            gradif_router_dim=cfg.gradif_router_dim,
            gradif_top_k=cfg.gradif_top_k,
            gradif_routing_mode=cfg.gradif_routing_mode,
            gradif_expert_bank_type=cfg.gradif_expert_bank_type,
            gradif_temperature=cfg.gradif_temperature,
            gradif_modality_scale=cfg.gradif_modality_scale,
            gradif_router_strategy=cfg.gradif_router_strategy,
            gradif_router_task_residual_scale=cfg.gradif_router_task_residual_scale,
            gradif_router_modality_residual_scale=cfg.gradif_router_modality_residual_scale,
            gradif_router_attention_pool=cfg.gradif_router_attention_pool,
            gradif_router_alpha_content_init=cfg.gradif_router_alpha_content_init,
            gradif_router_alpha_task_init=cfg.gradif_router_alpha_task_init,
            gradif_router_alpha_modality_init=cfg.gradif_router_alpha_modality_init,
            gradif_router_task_specific_gates=cfg.gradif_router_task_specific_gates,
            gradif_router_modality_specific_gates=cfg.gradif_router_modality_specific_gates,
            gradif_router_modality_content_adapters=cfg.gradif_router_modality_content_adapters,
            gradif_router_branch_hidden=cfg.gradif_router_branch_hidden,
            mofx_router_configuration=cfg.mofx_router_configuration,
            gradif_router_mi_min_group_size=cfg.gradif_router_mi_min_group_size,
            gradif_router_mi_queue_size=cfg.gradif_router_mi_queue_size,
            gradif_router_mi_current_fraction=cfg.gradif_router_mi_current_fraction,
            gradif_gamma_init=cfg.gradif_gamma_init,
            gradif_gamma_min=cfg.gradif_gamma_min,
            gradif_gamma_max=cfg.gradif_gamma_max,
            gradif_expert_scale_init=cfg.gradif_expert_scale_init,
            gradif_expert_scale_max=cfg.gradif_expert_scale_max,
            gradif_generic_mlp_bottleneck=cfg.gradif_generic_mlp_bottleneck,
            gradif_include_generalist=cfg.gradif_include_generalist,
            gradif_include_null_expert=cfg.gradif_include_null_expert,
            gradif_use_granularity_prior=cfg.gradif_use_granularity_prior,
            gradif_decomposition=cfg.gradif_decomposition,
            gradif_active_stages=tuple(cfg.gradif_active_stages),
            num_modalities=max(cfg.modality_map.values()) + 1,
            ultrasound_modality_id=ultrasound_id,
            class_names=cfg.class_names,
            seg_num_classes=len(cfg.segmentation_class_names),
            seg_img_size=int(cfg.target_size[0]),
            det_num_classes=len(cfg.detection_class_names),
            dino_repo_path=cfg.dino_repo_path,
            dino_cfg_path=cfg.dino_cfg_path,
            dino_pretrained_path=cfg.dino_pretrained_path,
            use_official_dino=True,
            use_cross_roi_reasoning=True,
            llm_model_name=cfg.llm_model_name,
            shared_head_seed=cfg.shared_head_seed,
            segmentation_gradif_fusion_init=cfg.segmentation_gradif_fusion_init,
            segmentation_gradif_max_delta_ratios=tuple(cfg.segmentation_gradif_max_delta_ratios),
            classification_gradif_fusion_init=cfg.classification_gradif_fusion_init,
            classification_gradif_max_delta_ratio=cfg.classification_gradif_max_delta_ratio,
        )
        return model

    def _validate_label_configuration(self) -> None:
        if not self.config.class_names:
            raise ValueError("class_names is empty")
        if not self.config.segmentation_class_names:
            raise ValueError("segmentation_class_names is empty")
        for modality, names in self.config.segmentation_classes_by_modality.items():
            _resolve_named_indices(self.config.segmentation_class_names, names,
                                   f"segmentation_classes_by_modality[{modality!r}]")
        for modality, names in self.config.classification_classes_by_modality.items():
            _resolve_named_indices(self.config.class_names, names,
                                   f"classification_classes_by_modality[{modality!r}]")
        invalid = [i for i in self.config.severity_class_indices if i < 0 or i >= len(self.config.class_names)]
        if invalid:
            raise ValueError(f"Invalid severity_class_indices: {invalid}")

    def _modality_id(self, modality: str, allow_missing: bool = False) -> Optional[int]:
        modality = _normalize_modality(modality)
        normalized = {_normalize_modality(k): int(v) for k, v in self.config.modality_map.items()}
        if modality not in normalized:
            if allow_missing:
                return None
            raise KeyError(f"Modality {modality} is missing from modality_map")
        return normalized[modality]

    def _prepare(self, image_path: str | Path) -> Tuple[torch.Tensor, np.ndarray]:
        image = _pil_rgb(image_path)
        original = np.asarray(image)
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor, original

    @staticmethod
    def _task_mask(execution_tasks: Set[str], modality: str, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((1, 4), dtype=torch.float32, device=device)
        if "segmentation" in execution_tasks:
            mask[0, 0] = 1.0
        if execution_tasks & {"classification", "birads", "pathology", "severity"}:
            mask[0, 1] = 1.0
        if "report" in execution_tasks:
            mask[0, 2] = 1.0
        if "detection" in execution_tasks:
            mask[0, 3] = 1.0
        return mask

    def infer(self, request: InferenceRequest) -> Dict[str, Any]:
        modality = _normalize_modality(request.modality)
        requested, execution = resolve_dependencies(
            modality,
            _normalize_tasks(request.tasks),
        )
        request_id = _safe_id(request.request_id)
        root = (self.output_root / request_id).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=False)

        image_path = Path(request.image_path).expanduser().resolve(strict=True)
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image is not a file: {image_path}")

        image_tensor, original_rgb = self._prepare(image_path)
        input_dir = root / "input"
        input_dir.mkdir(exist_ok=True)
        input_copy = (input_dir / image_path.name).resolve(strict=False)
        if request.save_input_copy:
            shutil.copy2(image_path, input_copy)

        modality_id = int(self._modality_id(modality))
        mod_ids = torch.tensor([modality_id], dtype=torch.long, device=self.device)
        task_mask = self._task_mask(execution, modality, self.device)
        is_lss_mri = torch.tensor([modality == "mri"], dtype=torch.bool, device=self.device)
        has_birads = torch.tensor([modality == "ultrasound" and "birads" in execution], dtype=torch.bool, device=self.device)
        has_pathology = torch.tensor([modality == "ultrasound" and "pathology" in execution], dtype=torch.bool, device=self.device)

        need_gradcam = modality == "ct" and "classification" in execution
        grad_context = torch.enable_grad() if need_gradcam else torch.no_grad()
        with grad_context:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.config.use_amp and self.device.type == "cuda",
            ):
                outputs = self.model(
                    image_tensor,
                    modality_ids=mod_ids,
                    task_mask=task_mask,
                    boxes=None,
                    det_targets=None,
                    captions=None,
                    report_labels=None,
                    use_pred_boxes_for_mri_cls=False,
                    det_score_thresh=self.config.detection_score_threshold,
                    is_lss_mri=is_lss_mri,
                    measurements=None,
                    measurements_present=None,
                    has_birads=has_birads,
                    has_pathology=has_pathology,
                    seg_mask_gt=None,
                    gt_birads_idx=None,
                    gt_pathology_idx=None,
                    compute_gradif_reference=False,
                    return_feats=need_gradcam,
                )

        result: Dict[str, Any] = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "request_root": str(root),
            "output_root": str(self.output_root),
            "input": {
                "source_image": str(image_path),
                "saved_image": str(input_copy) if request.save_input_copy else None,
                "modality": modality,
                "modality_id": modality_id,
                "requested_tasks": sorted(requested),
                "executed_tasks": sorted(execution),
                "task_mask": task_mask[0].detach().cpu().tolist(),
            },
            "outputs": {},
        }

        # Segmentation: decode only channels valid for the selected modality.
        segmentation_mask = None
        if "segmentation" in execution and outputs.get("pred_masks") is not None:
            active_seg_names = self.config.segmentation_classes_by_modality.get(modality, [])
            if not active_seg_names:
                raise ValueError(f"No segmentation_classes_by_modality entry for '{modality}'")
            seg_dir = root / "segmentation"
            seg_dir.mkdir(exist_ok=True)
            decoded = _modality_segmentation_from_logits(
                outputs["pred_masks"], self.config.segmentation_class_names,
                active_seg_names, self.config.segmentation_threshold,
                original_rgb.shape[:2],
            )
            segmentation_mask = decoded["label_map"]
            confidence_path = seg_dir / "confidence.npy"
            mask_path = seg_dir / "mask.png"
            overlay_path = seg_dir / "overlay.png"
            np.save(confidence_path, decoded["confidence"])
            Image.fromarray(segmentation_mask.astype(np.uint8)).save(mask_path)
            _save_rgb(overlay_path, _segmentation_overlay(original_rgb, segmentation_mask))
            class_outputs = []
            for local_id, (name, global_index) in enumerate(zip(decoded["class_names"], decoded["global_indices"]), start=1):
                safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                prob_path = seg_dir / f"probability_{safe_name}.npy"
                binary_path = seg_dir / f"mask_{safe_name}.png"
                np.save(prob_path, decoded["probabilities"][local_id - 1])
                Image.fromarray((decoded["binary_masks"][local_id - 1] * 255).astype(np.uint8)).save(binary_path)
                class_outputs.append({"local_label_id": local_id, "global_channel_index": int(global_index),
                                      "class_name": name, "probability_path": str(prob_path),
                                      "binary_mask_path": str(binary_path)})
            result["outputs"]["segmentation"] = {
                "active_classes": list(active_seg_names), "classes": class_outputs,
                "confidence_path": str(confidence_path), "mask_path": str(mask_path),
                "overlay_path": str(overlay_path), "threshold": self.config.segmentation_threshold,
            }
            if modality == "ultrasound":
                if "lesion" not in decoded["class_names"]:
                    raise ValueError("Ultrasound morphology requires the 'lesion' segmentation class")
                lesion_local = decoded["class_names"].index("lesion")
                lesion_mask = decoded["binary_masks"][lesion_local].astype(np.uint8)
                morph_dir = root / "ultrasound_morphology"
                morph_dir.mkdir(exist_ok=True)
                measurements = compute_ultrasound_morphology(lesion_mask)
                overlay = morph_dir / "measurements_overlay.png"
                values = morph_dir / "measurements.json"
                save_ultrasound_morphology_overlay(original_rgb, lesion_mask, measurements, overlay)
                serial_measurements = {k: v for k, v in measurements.items() if k != "ellipse"}
                _write_json(values, serial_measurements)
                result["outputs"]["ultrasound_morphology"] = {
                    "source_segmentation_class": "lesion",
                    "source_global_channel_index": int(decoded["global_indices"][lesion_local]),
                    "measurements": serial_measurements, "measurements_json": str(values),
                    "overlay_path": str(overlay),
                }

        # Generic classification: interpret only classes valid for this modality.
        class_logits = outputs.get("class_logits")
        if "classification" in execution and class_logits is not None:
            active_class_names = self.config.classification_classes_by_modality.get(modality, [])
            if not active_class_names:
                raise ValueError(f"No classification_classes_by_modality entry for '{modality}'")
            active_indices = _resolve_named_indices(self.config.class_names, active_class_names,
                                                    "classification_classes_by_modality")
            all_probs = torch.sigmoid(class_logits[0].float()).detach().cpu()
            predictions = []
            for label, idx in zip(active_class_names, active_indices):
                prob = float(all_probs[idx].item())
                predictions.append({"global_class_id": int(idx), "label": label,
                                    "probability": prob,
                                    "positive": bool(prob >= self.config.classification_threshold)})
            abnormal_positive = [p["label"] for p in predictions if p["positive"] and
                                 p["label"].lower().replace("_", " ") not in {"no finding", "normal"}]
            no_finding_positive = any(p["positive"] and p["label"].lower().replace("_", " ")
                                      in {"no finding", "normal"} for p in predictions)
            positive_labels = abnormal_positive or (["No_Finding"] if no_finding_positive else [])
            result["outputs"]["classification"] = {
                "active_classes": list(active_class_names), "active_global_indices": active_indices,
                "threshold": self.config.classification_threshold,
                "predictions": predictions, "positive_labels": positive_labels,
            }

        # Dedicated ultrasound heads.
        if modality == "ultrasound":
            birads_logits = outputs.get("birads_logits")
            if "birads" in execution and birads_logits is not None:
                cumulative = torch.sigmoid(birads_logits[0].float()).detach().cpu()
                thresholds = self.config.birads_thresholds or [0.5] * int(cumulative.numel())
                category_index = int(sum(float(p) > float(t) for p, t in zip(cumulative, thresholds)))
                category_index = min(category_index, len(self.config.birads_class_names) - 1)
                result["outputs"]["birads"] = {
                    "class_id": category_index,
                    "category": self.config.birads_class_names[category_index],
                    "cumulative_probabilities": cumulative.tolist(),
                    "thresholds": thresholds,
                }
            pathology_logits = outputs.get("pathology_meas_logits")
            if "pathology" in execution and pathology_logits is not None:
                probs = torch.softmax(pathology_logits[0].float(), dim=0).detach().cpu()
                pred = int(probs.argmax().item())
                result["outputs"]["pathology"] = {
                    "class_id": pred,
                    "label": self.config.pathology_class_names[pred],
                    "probabilities": {
                        self.config.pathology_class_names[i]: float(v)
                        for i, v in enumerate(probs.tolist())
                    },
                }

        # MRI detection and crop-based severity classification.
        mri_tags: List[str] = []
        if modality == "mri" and "detection" in execution:
            model_hw = tuple(int(v) for v in image_tensor.shape[-2:])
            original_hw = tuple(int(v) for v in original_rgb.shape[:2])

            # Detection is decoded in model-input coordinates because the ROI
            # crop classifier consumes the resized model tensor.
            model_boxes, labels, scores = self._decode_mri_detection(
                outputs,
                model_hw,
            )

            # JSON and overlay boxes are returned in original-image pixels.
            original_boxes = _scale_boxes_xyxy(
                model_boxes,
                from_hw=model_hw,
                to_hw=original_hw,
            )

            det_dir = (root / "detection").resolve(strict=False)
            det_dir.mkdir(exist_ok=True)

            severity_rows_all: List[Dict[str, Any]] = []
            if "severity" in execution and model_boxes.numel() > 0:
                severity_rows_all = self._infer_mri_severity(
                    image_tensor=image_tensor,
                    boxes=model_boxes,
                    labels=labels,
                    scores=scores,
                    output_dir=det_dir,
                    original_hw=original_hw,
                )

            severity_by_detection = _severity_map_by_detection(
                severity_rows_all
            )
            detected_level_rows = _best_severity_per_level(
                severity_rows_all
            )
            level_rows = _complete_mri_level_rows(
                detected_level_rows,
                self.config.detection_class_names,
            )

            for row in detected_level_rows:
                if row["label"].lower() not in {
                    "normal", "no_finding", "no finding"
                }:
                    mri_tags.append(f"{row['level']} {row['label']}")

            overlay = _draw_detection_overlay(
                original_rgb,
                original_boxes.detach().cpu(),
                labels.detach().cpu(),
                scores.detach().cpu(),
                self.config.detection_class_names,
                severity_by_detection or None,
            )
            overlay_path = (
                det_dir / "detection_overlay.png"
            ).resolve(strict=False)
            _save_rgb(overlay_path, overlay)

            detections: List[Dict[str, Any]] = []
            for detection_index in range(int(model_boxes.shape[0])):
                level_id = int(labels[detection_index].item())
                detection_row: Dict[str, Any] = {
                    "detection_index": detection_index,
                    "box_xyxy": original_boxes[detection_index]
                    .detach().cpu().tolist(),
                    "box_xyxy_model": model_boxes[detection_index]
                    .detach().cpu().tolist(),
                    "coordinate_space": "original_image_pixels",
                    "level_id": level_id,
                    "level": self.config.detection_class_names[level_id],
                    "score": float(scores[detection_index].item()),
                }

                severity_row = severity_by_detection.get(detection_index)
                if severity_row is not None:
                    detection_row["severity"] = {
                        "severity_id": severity_row["severity_id"],
                        "label": severity_row["label"],
                        "probabilities": severity_row["probabilities"],
                        "crop_box_xyxy": severity_row["crop_box_xyxy"],
                        "crop_path": severity_row["crop_path"],
                    }

                detections.append(detection_row)

            result["outputs"]["detection"] = {
                "num_detections": len(detections),
                "detections": detections,
                "coordinate_space": "original_image_pixels",
                "image_size": {
                    "height": original_hw[0],
                    "width": original_hw[1],
                },
                "overlay_path": str(overlay_path),
                "overlay_contains": (
                    ["level", "detection_score", "severity"]
                    if "severity" in execution
                    else ["level", "detection_score"]
                ),
            }

            # Return severity even when no boxes survive thresholding. This
            # keeps the response schema stable for clients.
            if "severity" in execution:
                result["outputs"]["severity"] = {
                    "num_configured_levels": len(
                        self.config.detection_class_names
                    ),
                    "num_detected_levels": len(detected_level_rows),
                    "levels": level_rows,
                    "severity_class_names": list(
                        self.config.severity_class_names
                    ),
                    "source": "detector_boxes_to_roi_crop_classifier",
                    "duplicate_level_policy": (
                        "highest_detection_score_per_level"
                    ),
                    "missing_level_policy": (
                        "return_status_not_detected_with_null_severity"
                    ),
                }

        # CT Grad-CAM uses the same forward graph as classification.
        if need_gradcam and class_logits is not None:
            ct_names = self.config.classification_classes_by_modality.get("ct", [])
            ct_indices = _resolve_named_indices(self.config.class_names, ct_names,
                                                "classification_classes_by_modality['ct']")
            if not ct_indices:
                raise ValueError("CT Grad-CAM requires configured CT classification classes")
            ct_probs = torch.sigmoid(class_logits[0, ct_indices].float())
            target_index = int(ct_indices[int(ct_probs.argmax().item())])
            cam_paths = _save_gradcam(
                original_rgb,
                outputs.get("ct_gradcam_feature", outputs.get("feat_c4_cls")),
                class_logits,
                target_index,
                root / "ct_gradcam",
            )
            target_label = self.config.class_names[target_index] if target_index < len(self.config.class_names) else str(target_index)
            result["outputs"]["ct_gradcam"] = {
                "target_class_id": target_index,
                "target_label": target_label,
                **cam_paths,
            }

        # Report generation is last so it can use earlier predicted findings.
        if "report" in requested:
            report_dir = root / "report"
            report_dir.mkdir(exist_ok=True)
            prompt = self._build_report_prompt(modality, result, mri_tags)
            text = self._generate_report(outputs, prompt, modality)
            report_path = report_dir / "report.txt"
            report_path.write_text(text, encoding="utf-8")
            prompt_path = report_dir / "prompt.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            result["outputs"]["report"] = {
                "text": text,
                "report_path": str(report_path),
                "prompt_path": str(prompt_path),
            }

        result_path = (root / "result.json").resolve(strict=False)
        result["result_json"] = str(result_path)
        _write_json(result_path, result)
        return result

    def _decode_mri_detection(
        self, outputs: Dict[str, Any], image_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if outputs.get("mri_pred_boxes") is not None:
            return (
                outputs["mri_pred_boxes"][0],
                outputs["mri_pred_labels"][0],
                outputs["mri_pred_scores"][0],
            )
        det = outputs.get("detection_active") or outputs.get("detection")
        if det is None:
            empty_boxes = torch.zeros((0, 4), device=self.device)
            empty = torch.zeros((0,), device=self.device)
            return empty_boxes, empty.long(), empty
        boxes, labels, scores, _ = self.model.decode_mri_det_predictions(
            det,
            image_size=image_hw,
            score_thresh=self.config.detection_score_threshold,
            topk=self.config.detection_topk,
            nms_iou=self.config.detection_nms_iou,
        )
        return boxes[0], labels[0], scores[0]

    def _infer_mri_severity(
        self,
        image_tensor: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        output_dir: Path,
        original_hw: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """Classify severity for every valid detected MRI level ROI.

        The returned rows retain `detection_index`, preventing a skipped or
        invalid crop from shifting severity predictions onto the wrong box.
        """
        model_h, model_w = image_tensor.shape[-2:]
        crops: List[torch.Tensor] = []
        crop_meta: List[Tuple[int, List[int]]] = []

        for detection_index, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad_x = int(round(0.20 * bw))
            pad_y = int(round(0.20 * bh))

            crop_x1 = max(0, int(round(x1)) - pad_x)
            crop_y1 = max(0, int(round(y1)) - pad_y)
            crop_x2 = min(model_w, int(round(x2)) + pad_x)
            crop_y2 = min(model_h, int(round(y2)) + pad_y)

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = image_tensor[
                :, :, crop_y1:crop_y2, crop_x1:crop_x2
            ]
            if crop.shape[-2] < 2 or crop.shape[-1] < 2:
                continue

            crop = F.interpolate(
                crop,
                size=self.config.target_size,
                mode="bilinear",
                align_corners=False,
            )
            crops.append(crop[0])
            crop_meta.append(
                (
                    detection_index,
                    [crop_x1, crop_y1, crop_x2, crop_y2],
                )
            )

        if not crops:
            return []

        crop_batch = torch.stack(crops).to(self.device)
        num_crops = crop_batch.shape[0]
        mod_ids = torch.full(
            (num_crops,),
            int(self._modality_id("mri")),
            dtype=torch.long,
            device=self.device,
        )
        crop_task_mask = torch.zeros(
            (num_crops, 4),
            dtype=torch.float32,
            device=self.device,
        )
        crop_task_mask[:, 1] = 1.0
        crop_is_lss_mri = torch.zeros(
            (num_crops,),
            dtype=torch.bool,
            device=self.device,
        )

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.config.use_amp and self.device.type == "cuda",
        ):
            crop_outputs = self.model(
                crop_batch,
                modality_ids=mod_ids,
                task_mask=crop_task_mask,
                boxes=None,
                det_targets=None,
                captions=None,
                report_labels=None,
                use_pred_boxes_for_mri_cls=False,
                det_score_thresh=self.config.detection_score_threshold,
                is_lss_mri=crop_is_lss_mri,
            )

        logits = crop_outputs["class_logits"].float()
        if not self.config.severity_class_indices:
            raise ValueError(
                "severity_class_indices is required for MRI severity "
                "inference and must match the three crop-classifier columns "
                "used during test_multitask."
            )

        logits = logits[:, self.config.severity_class_indices]
        probs = torch.softmax(logits, dim=1).detach().cpu()
        pred_ids = probs.argmax(dim=1)

        crops_dir = (output_dir / "roi_crops").resolve(strict=False)
        crops_dir.mkdir(exist_ok=True)
        rows: List[Dict[str, Any]] = []

        for crop_index, (detection_index, crop_box_model) in enumerate(
            crop_meta
        ):
            level_id = int(labels[detection_index].item())
            severity_id = int(pred_ids[crop_index].item())
            level_name = self.config.detection_class_names[level_id]

            crop_rgb = _denormalize(crop_batch[crop_index])
            safe_level = re.sub(
                r"[^A-Za-z0-9_.-]+", "_", level_name
            )
            crop_path = (
                crops_dir
                / f"roi_{detection_index:02d}_{safe_level}.png"
            ).resolve(strict=False)
            _save_rgb(crop_path, crop_rgb)

            crop_box_tensor = torch.tensor(
                [crop_box_model],
                dtype=torch.float32,
                device=boxes.device,
            )
            crop_box_original = _scale_boxes_xyxy(
                crop_box_tensor,
                from_hw=(model_h, model_w),
                to_hw=original_hw,
            )[0]
            box_original = _scale_boxes_xyxy(
                boxes[detection_index:detection_index + 1],
                from_hw=(model_h, model_w),
                to_hw=original_hw,
            )[0]

            rows.append({
                "status": "detected",
                "detection_index": detection_index,
                "level_id": level_id,
                "level": level_name,
                "detection_score": float(scores[detection_index].item()),
                "box_xyxy": box_original.detach().cpu().tolist(),
                "box_xyxy_model": boxes[detection_index]
                .detach().cpu().tolist(),
                "severity_id": severity_id,
                "label": self.config.severity_class_names[severity_id],
                "probabilities": {
                    self.config.severity_class_names[j]: float(value)
                    for j, value in enumerate(
                        probs[crop_index].tolist()
                    )
                },
                "crop_box_xyxy": crop_box_original
                .detach().cpu().tolist(),
                "crop_box_xyxy_model": crop_box_model,
                "crop_path": str(crop_path),
            })

        return rows

    def _build_report_prompt(
        self,
        modality: str,
        result: Dict[str, Any],
        mri_tags: Sequence[str],
    ) -> str:
        """
        Build the same user-side prompt content used by test_multitask.

        X-ray / CT:
            Detected abnormalities: ... Report:
            or No abnormality detected. Report:

        MRI:
            Uses predicted level-severity tags in the same format.

        BUS-BRA ultrasound:
            Uses the same compact descriptor prompt:
            Breast ultrasound. BI-RADS: ... Pathology: ...
            Lesion W x H px. Orientation: ... Report:
        """
        outputs = result["outputs"]

        if modality == "ultrasound":
            birads = outputs.get("birads", {})
            pathology = outputs.get("pathology", {})
            morph = (
                outputs.get("ultrasound_morphology", {})
                .get("measurements", {})
            )

            descriptor: Dict[str, Any] = {}

            if birads:
                descriptor["birads"] = birads.get("category")

            if pathology:
                descriptor["pathology"] = pathology.get("label")

            max_diameter = morph.get("max_diameter_px")
            perpendicular_diameter = morph.get(
                "perpendicular_diameter_px"
            )
            bbox_width = morph.get("bbox_width_px")
            bbox_height = morph.get("bbox_height_px")

            if max_diameter is not None:
                descriptor["max_diameter_px"] = max_diameter

            if perpendicular_diameter is not None:
                descriptor[
                    "perpendicular_diameter_px"
                ] = perpendicular_diameter

            if bbox_width is not None and bbox_height is not None:
                bbox_width = float(bbox_width)
                bbox_height = float(bbox_height)

                if bbox_width > bbox_height:
                    descriptor["orientation"] = (
                        "wider-than-tall orientation"
                    )
                elif bbox_height > bbox_width:
                    descriptor["orientation"] = (
                        "taller-than-wide orientation"
                    )
                else:
                    descriptor["orientation"] = (
                        "approximately equal width and height"
                    )

            # Exact equivalent of dataset._build_descriptor_content().
            parts = ["Breast ultrasound."]

            birads_value = descriptor.get("birads")
            if (
                birads_value is not None
                and str(birads_value).strip() != ""
            ):
                parts.append(f"BI-RADS: {birads_value}.")

            pathology_value = descriptor.get("pathology")
            if pathology_value:
                parts.append(f"Pathology: {pathology_value}.")

            md = descriptor.get("max_diameter_px")
            pd = descriptor.get("perpendicular_diameter_px")

            try:
                if (
                    md
                    and pd
                    and float(md) > 0
                    and float(pd) > 0
                ):
                    parts.append(
                        f"Lesion {float(md):.1f} x "
                        f"{float(pd):.1f} px."
                    )
            except (TypeError, ValueError):
                pass

            orientation = descriptor.get("orientation")
            if orientation:
                parts.append(f"Orientation: {orientation}.")

            parts.append("Report:")
            return " ".join(parts)

        if modality == "mri":
            active_tags = [
                str(tag)
                for tag in mri_tags
                if str(tag).strip()
            ]
        else:
            active_tags = list(
                outputs.get("classification", {})
                .get("positive_labels", [])
            )

            # test_multitask removes the normal label before prompting.
            active_tags = [
                str(tag)
                for tag in active_tags
                if str(tag) != "No_Finding"
            ]

        # Exact equivalent of make_report_prompt_content(..., use_guidance=True).
        if active_tags:
            return (
                "Detected abnormalities: "
                + ", ".join(active_tags)
                + ". Report:"
            )

        return "No abnormality detected. Report:"

    def _generate_report(
        self,
        outputs: Dict[str, Any],
        prompt: str,
        modality: str,
    ) -> str:
        visual = outputs.get("visual_tokens")

        if visual is None:
            return (
                "Report generation could not run because visual "
                "report tokens were not produced."
            )

        # Exact modality names used by the training dataset modality map.
        modality_name = {
            "ct": "CT",
            "mri": "MRI",
            "ultrasound": "Ultrasound",
            "xray": "XRay",
        }[modality]

        # Exact test_multitask message structure and wording.
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {modality_name} radiologist. "
                    "Describe the findings."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        report_decoder = self.model.report_decoder
        original_max_len = getattr(
            report_decoder,
            "max_len",
            None,
        )

        try:
            if original_max_len is not None:
                report_decoder.max_len = int(
                    self.config.report_max_new_tokens
                )

            with torch.no_grad():
                generated = report_decoder.generate(
                    visual_features=visual,
                    prompt_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    # Same decoding settings as test_multitask.
                    do_sample=False,
                    num_beams=3,
                )
        finally:
            if original_max_len is not None:
                report_decoder.max_len = original_max_len

        # Same direct decoding and cleanup used by test_multitask.
        text = self.tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        if "Report:" in text:
            text = text.split("Report:")[-1].strip()
        elif "findings." in text[:20].lower():
            text = text.split("findings.")[-1].strip()

        return text


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified medical image inference")
    parser.add_argument("--config", required=True, help="Inference JSON configuration")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--modality", required=True, choices=["xray", "ct", "mri", "ultrasound"])
    parser.add_argument(
        "--tasks", required=True,
        help="Comma-separated tasks, e.g. segmentation,classification,report or all",
    )
    parser.add_argument("--request-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = InferenceConfig.from_json(args.config)
    engine = UnifiedInferenceEngine(config)
    request = InferenceRequest(
        image_path=args.image,
        modality=args.modality,
        tasks=[v.strip() for v in args.tasks.split(",") if v.strip()],
        request_id=args.request_id,
    )
    result = engine.infer(request)
    print(json.dumps(_json_ready(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

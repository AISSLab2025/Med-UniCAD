"""Checkpoint-compatible training pipeline for the final Med-UniCAD model.

The public entry point is :func:`medunicad.training.runner.train`. This module
contains the full data, loss, optimization, validation, and checkpoint flow
used by that runner.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import json
import os
import gc
import hashlib
import glob
import math
import re
import time
import platform
from contextlib import contextmanager
from pathlib import Path
import seaborn as sns
import random
from collections import defaultdict
import cv2
from sklearn.metrics import classification_report
from torchvision.ops import batched_nms
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, hamming_loss,
    accuracy_score, average_precision_score, precision_score,
    recall_score, balanced_accuracy_score, cohen_kappa_score,
    precision_recall_fscore_support,
)
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from torch.utils.data import Sampler
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import label_binarize
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer
from peft import get_peft_model
import torch.nn.functional as F
from medunicad.modeling._components import SwinTransformer
from medunicad.modeling.freqmoe_dense import MoFXDense as SwinMultiTask
from medunicad.modeling._components import (
    SwinUnetDecoderSys,
    MEASUREMENT_KEYS_BUSBRA, N_MEAS_BUSBRA, REPORT_MORPHOLOGY_KEYS_BUSBRA,
)
from medunicad.losses.dino_matcher import HungarianMatcher
from medunicad.losses.dino_criterion import DINOCriterion
from medunicad.losses.box_ops import *

def set_reproducibility(seed=42, deterministic=True):
    """Seed model and data operations for reproducible paper runs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not bool(deterministic)
    torch.backends.cudnn.deterministic = bool(deterministic)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    except Exception:
        pass


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# -----------------------------------------------------------------------------
# 1. Experiment Manager (Logging, Saving, Plotting)
# -----------------------------------------------------------------------------
class ExperimentManager:
    def __init__(self, base_dir="outputs/runs", resume_run_id=None):
        if resume_run_id:
            self.run_dir = os.path.join(base_dir, resume_run_id)
            if not os.path.exists(self.run_dir):
                self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.run_dir = os.path.join(base_dir, self.timestamp)
        else:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.run_dir = os.path.join(base_dir, self.timestamp)
            
        # Base Folders
        self.ckpt_dir = os.path.join(self.run_dir, "checkpoints")
        self.base_sample_dir = os.path.join(self.run_dir, "test_samples_seg")
        self.base_report_dir = os.path.join(self.run_dir, "test_samples_report")
        self.base_cls_dir = os.path.join(self.run_dir, "test_samples_cls") # NEW
        self.debug_dir = os.path.join(self.run_dir, "debug_vis")
        
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.base_sample_dir, exist_ok=True)
        os.makedirs(self.base_report_dir, exist_ok=True)
        os.makedirs(self.base_cls_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.run_dir, "training_log.xlsx")
        self.history = []
        self.best_val_loss = float('inf')
        self.best_ultrasound_score = -float('inf')
        self.best_model_path = os.path.join(self.ckpt_dir, "best_model.pth")
        if resume_run_id and os.path.isfile(self.log_file):
            try:
                df = pd.read_excel(self.log_file)
                # keep only real epoch rows
                if "Epoch" in df.columns:
                    df["Epoch_Num"] = pd.to_numeric(df["Epoch"], errors="coerce")
                    df = df.dropna(subset=["Epoch_Num"]).sort_values("Epoch_Num")
                    df = df.drop(columns=["Epoch_Num"], errors="ignore")
                self.history = df.to_dict("records")

                # also restore best_val_loss from file (optional but nice)
                val_cols = [c for c in df.columns if c.lower() in ["val_loss", "Val_loss".lower()]]
                if "Val_loss" in df.columns and len(df) > 0:
                    self.best_val_loss = float(df["Val_loss"].min())
                if "Val_UltrasoundScore" in df.columns and len(df) > 0:
                    _us_values = pd.to_numeric(
                        df["Val_UltrasoundScore"], errors="coerce"
                    ).dropna()
                    if len(_us_values) > 0:
                        self.best_ultrasound_score = float(_us_values.max())
                print(f"[Resume] Loaded {len(self.history)} history rows from {self.log_file}")
            except Exception as e:
                print(f"[Resume] Failed to load history: {e}")
        self.saved_counts = defaultdict(int)
        
    def _prepare_image(self, img_tensor):
        img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        return np.clip(img_np, 0, 1)
    
    def save_seg_sample(self, img_tensor, pred_mask, gt_mask, dice_score, idx, ds_name):
        try:
            # 1. Check for "Boring" Samples (True Negatives)
            # GT is on GPU, move to CPU for check
            is_gt_empty = gt_mask.sum().item() == 0
            
            # Pred is logits, so check sigmoid > 0.5
            if pred_mask is not None:
                is_pred_empty = (torch.sigmoid(pred_mask) > 0.5).sum().item() == 0
            else:
                is_pred_empty = True

            # SKIP if both are empty (Correctly predicted nothing)
            if is_gt_empty and is_pred_empty:
                return 

            # ---------------------------------------------------
            
            # Create Dataset Subfolder
            ds_folder = os.path.join(self.base_sample_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)
            
            # Save every 5th *interesting* image
            count_key = f"seg_{ds_name}"
            # self.saved_counts[count_key] += 1
            # if self.saved_counts[count_key] % 5 != 0: return
            self.saved_counts[count_key] += 1

            img_np = self._prepare_image(img_tensor)

            # Helper for color map
            def to_cmap(mask):
                if mask is None: return None
                mask_np = mask.detach().cpu().numpy()
                if mask_np.shape[0] == 1:
                    return (mask_np[0] > 0.5).astype(np.float32)
                else:
                    if mask_np.max() <= 1.0: is_bg = np.max(mask_np, axis=0) < 0.5
                    else: is_bg = np.max(mask_np, axis=0) == 0
                    class_map = np.argmax(mask_np, axis=0) + 1
                    class_map[is_bg] = 0
                    return class_map

            if pred_mask is not None:
                pred_prob = torch.sigmoid(pred_mask)
                pred_map = to_cmap(pred_prob)
            else: pred_map = None
            gt_map = to_cmap(gt_mask)

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"Sample {idx} | Dice: {dice_score:.4f}", fontsize=10)
            
            plt.subplot(1, 3, 1); plt.imshow(img_np); plt.title("Input"); plt.axis('off')
            plt.subplot(1, 3, 2); plt.imshow(img_np, cmap='gray')
            if gt_map is not None:
                masked_gt = np.ma.masked_where(gt_map == 0, gt_map)
                plt.imshow(masked_gt, cmap='tab20', alpha=0.7, vmin=1, vmax=20)
            plt.title("GT"); plt.axis('off')
            plt.subplot(1, 3, 3); plt.imshow(img_np, cmap='gray')
            if pred_map is not None:
                masked_pred = np.ma.masked_where(pred_map == 0, pred_map)
                plt.imshow(masked_pred, cmap='tab20', alpha=0.7, vmin=1, vmax=20)
            plt.title("Pred"); plt.axis('off')
            
            plt.savefig(os.path.join(ds_folder, f"seg_{idx}.png"))
            plt.close()
        except Exception as e: print(f"Seg Save Error: {e}")

    # --- SAVE CLASSIFICATION ---
    def save_cls_sample(self, img_tensor, gt_labels, pred_probs, class_names, idx, ds_name):
        try:
            ds_folder = os.path.join(self.base_cls_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)
            
            count_key = f"cls_{ds_name}"
            # self.saved_counts[count_key] += 1
            # if self.saved_counts[count_key] % 10 != 0: return
            self.saved_counts[count_key] += 1

            img_np = self._prepare_image(img_tensor)
            
            # --- FIX: Consistent Naming ---
            # Define what we call an empty label vector (usually "No_Finding" matches your class list)
            NEGATIVE_CLASS_NAME = "No_Finding" 

            # 1. Format Ground Truth
            gt_indices = np.where(gt_labels.cpu().numpy() == 1)[0]
            if len(gt_indices) > 0:
                gt_str = ", ".join([class_names[i] for i in gt_indices])
            else:
                gt_str = NEGATIVE_CLASS_NAME  # Fallback matches the class name
            
            # 2. Format Prediction
            pred_indices = np.where(pred_probs.cpu().numpy() > 0.5)[0]
            if len(pred_indices) > 0:
                pred_str = ", ".join([f"{class_names[i]} ({pred_probs[i]:.2f})" for i in pred_indices])
            else:
                pred_str = f"{NEGATIVE_CLASS_NAME} (Low Probs)" # Fallback matches the class name
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img_np)
            plt.axis('off')
            
            # --- FIX: Text Positioning ---
            # Use 'figure' coordinates (0 to 1) instead of data coordinates (pixels)
            # This ensures the text is always visible at the top-left, regardless of image size
            text = f"GT: {gt_str}\nPred: {pred_str}"
            plt.text(0.02, 0.98, text, 
                     color='white', fontsize=11, weight='bold',
                     verticalalignment='top', horizontalalignment='left',
                     transform=plt.gca().transAxes, # <--- Key change: Relative to axes
                     bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))
            
            plt.savefig(os.path.join(ds_folder, f"cls_{idx}.png"), bbox_inches='tight')
            plt.close()
        except Exception as e: 
            print(f"Cls Save Error: {e}")

    # --- SAVE REPORT ---
    # def save_rep_sample(self, img_tensor, gt_text, pred_text, idx, ds_name, scores):
    #     try:
    #         ds_folder = os.path.join(self.base_report_dir, ds_name)
    #         os.makedirs(ds_folder, exist_ok=True)
            
    #         count_key = f"rep_{ds_name}"
    #         # self.saved_counts[count_key] += 1
    #         # if self.saved_counts[count_key] % 10 != 0: return
    #         self.saved_counts[count_key] += 1

    #         img_np = self._prepare_image(img_tensor)

    #         plt.figure(figsize=(10, 10))
    #         plt.subplot(2, 1, 1); plt.imshow(img_np); plt.axis('off'); plt.title(f"Sample {idx}")
            
    #         plt.subplot(2, 1, 2); plt.axis('off')
    #         wrapped_gt = "\n".join(textwrap.wrap(f"GT: {gt_text}", width=60))
    #         wrapped_pred = "\n".join(textwrap.wrap(f"PRED: {pred_text}", width=60))
    #         score_str = f"METEOR: {scores.get('meteor',0):.3f} | ROUGE-L: {scores.get('rouge',0):.3f}"
            
    #         plt.text(0.01, 0.95, f"{wrapped_gt}\n\n{wrapped_pred}\n\n{score_str}", fontsize=11, verticalalignment='top')
            
    #         plt.savefig(os.path.join(ds_folder, f"rep_{idx}.png"))
    #         plt.close()
    #     except Exception as e: print(f"Rep Save Error: {e}")
    def save_rep_record(self, image_path, gt_text, pred_text, idx, ds_name, scores, extra=None):
        """
        Saves every report to a JSONL file (one line per sample).
        Much lighter than saving PNG for everything.
        """
        try:
            ds_folder = os.path.join(self.base_report_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)

            out_path = os.path.join(ds_folder, "all_reports.jsonl")

            rec = {
                "idx": int(idx),
                "image_path": str(image_path),
                "gt_report": gt_text,
                "pred_report": pred_text,
                "bleu": float(scores.get("bleu", 0.0)),
                "meteor": float(scores.get("meteor", 0.0)),
                "rougeL": float(scores.get("rouge", 0.0)),
            }
            if extra:
                rec.update(extra)

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        except Exception as e:
            print(f"Rep Record Save Error: {e}")

    def save_det_sample(self, img_tensor, pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels, idx, ds_name, det_class_names=None):
        try:
            base_det_dir = os.path.join(self.run_dir, "test_samples_det")
            ds_folder = os.path.join(base_det_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)

            count_key = f"det_{ds_name}"
            self.saved_counts[count_key] += 1

            # Skip boring samples: no GT and no pred
            if (gt_boxes is None or gt_boxes.numel() == 0) and (pred_boxes is None or pred_boxes.numel() == 0):
                return

            img_np = self._prepare_image(img_tensor)

            plt.figure(figsize=(12, 6))

            # GT
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("GT Detection")
            plt.axis("off")

            if gt_boxes is not None and gt_boxes.numel() > 0:
                gt_boxes_np = gt_boxes.detach().cpu().numpy()
                gt_labels_np = gt_labels.detach().cpu().numpy() if gt_labels is not None and gt_labels.numel() > 0 else None

                for j, box in enumerate(gt_boxes_np):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
                    plt.gca().add_patch(rect)

                    if gt_labels_np is not None:
                        lbl = int(gt_labels_np[j])
                        lbl_txt = det_class_names[lbl] if det_class_names is not None and lbl < len(det_class_names) else str(lbl)
                        plt.text(
                            x1, max(0, y1 - 4), f"GT:{lbl_txt}",
                            color='yellow', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.6, pad=1)
                        )

            # Pred
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            plt.title("Pred Detection")
            plt.axis("off")

            if pred_boxes is not None and pred_boxes.numel() > 0:
                pred_boxes_np = pred_boxes.detach().cpu().numpy()
                pred_labels_np = pred_labels.detach().cpu().numpy() if pred_labels is not None and pred_labels.numel() > 0 else None
                pred_scores_np = pred_scores.detach().cpu().numpy() if pred_scores is not None and pred_scores.numel() > 0 else None

                for j, box in enumerate(pred_boxes_np):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
                    plt.gca().add_patch(rect)

                    txt = "Pred"
                    if pred_labels_np is not None:
                        lbl = int(pred_labels_np[j])
                        lbl_txt = det_class_names[lbl] if det_class_names is not None and lbl < len(det_class_names) else str(lbl)
                        txt = f"P:{lbl_txt}"
                    if pred_scores_np is not None:
                        txt += f" ({pred_scores_np[j]:.2f})"

                    plt.text(
                        x1, max(0, y1 - 4), txt,
                        color='cyan', fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.6, pad=1)
                    )

            plt.tight_layout()
            plt.savefig(os.path.join(ds_folder, f"det_{idx}.png"), bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Det Save Error: {e}")


    def save_mri_roi_sample(self, img_tensor, pred_boxes, pred_logits, gt_boxes, gt_severity,
                            idx, ds_name, severity_names=None):
        try:
            base_roi_dir = os.path.join(self.run_dir, "test_samples_mri_roi")
            ds_folder = os.path.join(base_roi_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)

            count_key = f"mriroi_{ds_name}"
            self.saved_counts[count_key] += 1

            if (gt_boxes is None or gt_boxes.numel() == 0) and (pred_boxes is None or pred_boxes.numel() == 0):
                return

            img_np = self._prepare_image(img_tensor)

            plt.figure(figsize=(12, 6))

            # GT severity
            plt.subplot(1, 2, 1)
            plt.imshow(img_np)
            plt.title("GT MRI ROI Severity")
            plt.axis("off")

            if gt_boxes is not None and gt_boxes.numel() > 0:
                gt_boxes_np = gt_boxes.detach().cpu().numpy()
                gt_sev_np = gt_severity.detach().cpu().numpy() if gt_severity is not None and gt_severity.numel() > 0 else None

                for j, box in enumerate(gt_boxes_np):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
                    plt.gca().add_patch(rect)

                    if gt_sev_np is not None and j < len(gt_sev_np):
                        sev = int(gt_sev_np[j])
                        sev_txt = severity_names[sev] if severity_names is not None and sev < len(severity_names) else str(sev)
                        plt.text(
                            x1, max(0, y1 - 4), f"GT:{sev_txt}",
                            color='yellow', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.6, pad=1)
                        )

            # Pred severity
            plt.subplot(1, 2, 2)
            plt.imshow(img_np)
            plt.title("Pred MRI ROI Severity")
            plt.axis("off")

            if pred_boxes is not None and pred_boxes.numel() > 0:
                pred_boxes_np = pred_boxes.detach().cpu().numpy()
                pred_cls = pred_logits.argmax(dim=1).detach().cpu().numpy() if pred_logits is not None and pred_logits.numel() > 0 else None
                pred_prob = torch.softmax(pred_logits, dim=1).max(dim=1).values.detach().cpu().numpy() if pred_logits is not None and pred_logits.numel() > 0 else None

                for j, box in enumerate(pred_boxes_np):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
                    plt.gca().add_patch(rect)

                    if pred_cls is not None and j < len(pred_cls):
                        sev = int(pred_cls[j])
                        sev_txt = severity_names[sev] if severity_names is not None and sev < len(severity_names) else str(sev)
                        txt = f"P:{sev_txt}"
                        if pred_prob is not None and j < len(pred_prob):
                            txt += f" ({pred_prob[j]:.2f})"
                        plt.text(
                            x1, max(0, y1 - 4), txt,
                            color='cyan', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.6, pad=1)
                        )

            plt.tight_layout()
            plt.savefig(os.path.join(ds_folder, f"mri_roi_{idx}.png"), bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"MRI ROI Save Error: {e}")
            
    # --- PLOT CONFUSION MATRIX PER DATASET ---
    def plot_dataset_cm(self, y_true, y_probs, class_names, ds_name):
        try:
            ds_folder = os.path.join(self.base_cls_dir, ds_name)
            os.makedirs(ds_folder, exist_ok=True)
            
            # Flatten for global binary matrix (Present vs Absent)
            y_pred_bin = (y_probs > 0.5).astype(int).flatten()
            y_true_bin = y_true.astype(int).flatten()
            
            cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Absent', 'Present'], yticklabels=['Absent', 'Present'])
            plt.title(f"Confusion Matrix: {ds_name}")
            plt.savefig(os.path.join(ds_folder, f"confusion_matrix.png"))
            plt.close()
        except: pass

    def log_epoch(self, epoch, train_metrics, val_metrics):
        entry = {"Epoch": int(epoch), "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        for k, v in train_metrics.items(): entry[f"Train_{k}"] = v
        if val_metrics:
            for k, v in val_metrics.items(): entry[f"Val_{k}"] = v

        # --- upsert by epoch ---
        self.history = [r for r in self.history if int(r.get("Epoch", -1)) != int(epoch)]
        self.history.append(entry)

        df = pd.DataFrame(self.history).sort_values("Epoch")
        df.to_excel(self.log_file, index=False)
        self.plot_curves()

    def plot_curves(self):
        df = pd.DataFrame(self.history)
        
        # --- FIX STARTS HERE ---
        # 1. Force 'Epoch' to numeric. Any string like "TEST" becomes NaN (Not a Number)
        df['Epoch_Num'] = pd.to_numeric(df['Epoch'], errors='coerce')
        
        # 2. Remove rows where Epoch_Num is NaN (this drops the "TEST" row)
        df_plot = df.dropna(subset=['Epoch_Num']).sort_values('Epoch_Num')
        # --- FIX ENDS HERE ---

        # Identify metrics (loss, acc, seg_loss, etc.)
        metrics = [col.replace("Train_", "") for col in df_plot.columns if col.startswith("Train_")]
        
        # 1. Main Loss Plot
        plt.figure(figsize=(10, 6))
        
        # CRITICAL CHANGE: Use 'df_plot' and 'Epoch_Num' instead of 'df' and 'Epoch'
        plt.plot(df_plot['Epoch_Num'], df_plot['Train_loss'], label='Train Total', marker='o')
        
        if 'Val_loss' in df_plot.columns and not df_plot['Val_loss'].isnull().all():
            plt.plot(df_plot['Epoch_Num'], df_plot['Val_loss'], label='Val Total', marker='x')
            
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Total Loss Curve'); plt.legend(); plt.grid(True)
        plt.savefig(os.path.join(self.run_dir, "loss_total.png")); plt.close()

        # 2. Metric Plots (Accuracy, Sub-losses)
        for m in metrics:
            if m == 'loss': continue # Already plotted
            
            plt.figure(figsize=(10, 6))
            # CRITICAL CHANGE: Use 'df_plot' and 'Epoch_Num'
            plt.plot(df_plot['Epoch_Num'], df_plot[f'Train_{m}'], label=f'Train {m}', marker='o')
            
            val_key = f"Val_{m}"
            if val_key in df_plot.columns and not df_plot[val_key].isnull().all():
                plt.plot(df_plot['Epoch_Num'], df_plot[val_key], label=f'Val {m}', marker='x')
            
            plt.xlabel('Epoch')
            plt.ylabel(m.capitalize()) # Label y-axis
            plt.title(f'{m.capitalize()} Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.run_dir, f"curve_{m}.png")); plt.close()
    
    def plot_dataset_cm_masked(self, y_true, y_pred, y_mask, class_names, ds,
                           normalize: bool = False,
                           min_valid: int = 5,
                           save_name: str | None = None):
        """
        Mask-aware per-dataset confusion matrix plotter for multi-label classification.

        Interprets each class as an independent binary task and aggregates a 2x2 confusion
        matrix (TN, FP, FN, TP) per class using ONLY valid entries (y_mask==1).
        Then plots a compact grid of small 2x2 matrices (one per class).

        Args:
            y_true: np.ndarray [N, C] binary ground truth.
            y_pred: np.ndarray [N, C] binary predictions (already thresholded).
            y_mask: np.ndarray [N, C] {0,1} indicating which labels are valid for each sample/class.
            class_names: list[str] of length C.
            ds: dataset name (string) used in title + filename.
            normalize: if True, normalize each 2x2 matrix by its total valid count.
            min_valid: skip plotting classes with fewer than this many valid samples.
            save_name: optional explicit filename (png). If None, auto name used.

        Returns:
            dict with per-class 2x2 matrices + meta (useful for debugging/logging).
        """
        # ---- to numpy (defensive) ----
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        y_mask = np.asarray(y_mask).astype(int)

        assert y_true.shape == y_pred.shape == y_mask.shape, \
            f"Shape mismatch: true={y_true.shape}, pred={y_pred.shape}, mask={y_mask.shape}"

        N, C = y_true.shape

        # Make sure invalid entries do not contribute (safety)
        y_true_eff = y_true.copy()
        y_pred_eff = y_pred.copy()
        y_true_eff[y_mask == 0] = 0
        y_pred_eff[y_mask == 0] = 0

        per_class = []
        kept_indices = []

        for c in range(C):
            valid = (y_mask[:, c] == 1)
            n_valid = int(valid.sum())
            if n_valid < min_valid:
                continue

            yt = y_true_eff[valid, c]
            yp = y_pred_eff[valid, c]

            # Force binary (in case something weird slipped in)
            yt = (yt > 0).astype(int)
            yp = (yp > 0).astype(int)

            # Confusion matrix for binary labels
            # cm = [[TN, FP],
            #       [FN, TP]]
            cm = confusion_matrix(yt, yp, labels=[0, 1])

            if normalize:
                denom = cm.sum()
                cm = cm.astype(float) / (denom + 1e-9)

            per_class.append((class_names[c], cm, n_valid))
            kept_indices.append(c)

        # If nothing to plot, exit cleanly
        if len(per_class) == 0:
            print(f"[CM][{ds}] No classes had >= {min_valid} valid samples. Skipping CM plot.")
            return {"ds": ds, "min_valid": min_valid, "per_class": []}

        # ---- layout: grid of mini 2x2 matrices ----
        K = len(per_class)
        # heuristic grid size
        ncols = int(np.ceil(np.sqrt(K)))
        nrows = int(np.ceil(K / ncols))

        fig_w = max(10, 3.2 * ncols)
        fig_h = max(6, 3.0 * nrows)
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
        axes = np.array(axes).reshape(-1)

        for ax_i, ax in enumerate(axes):
            if ax_i >= K:
                ax.axis("off")
                continue

            name, cm, n_valid = per_class[ax_i]

            im = ax.imshow(cm, interpolation="nearest")
            ax.set_title(f"{name}\n(n={n_valid})", fontsize=9)

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["0", "1"], fontsize=8)
            ax.set_yticklabels(["0", "1"], fontsize=8)
            ax.set_xlabel("Pred", fontsize=8)
            ax.set_ylabel("True", fontsize=8)

            # annotate values
            for r in range(2):
                for c in range(2):
                    val = cm[r, c]
                    if normalize:
                        txt = f"{val:.2f}"
                    else:
                        txt = f"{int(val)}"
                    ax.text(c, r, txt, ha="center", va="center", fontsize=9)

        # Global title + colorbar
        title_norm = " (normalized)" if normalize else ""
        fig.suptitle(f"Masked Confusion Matrices{title_norm} — {ds}", fontsize=14)

        # Single colorbar (works even though each axis has its own imshow)
        fig.colorbar(axes[0].images[0], ax=axes.tolist(), fraction=0.02, pad=0.02)

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # ---- save ----
        out_dir = os.path.join(self.run_dir, "confusion_matrices")
        os.makedirs(out_dir, exist_ok=True)

        if save_name is None:
            save_name = f"cm_masked_{ds}{'_norm' if normalize else ''}.png"

        save_path = os.path.join(out_dir, save_name)
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

        print(f"[Saved][CM][{ds}] {save_path}")

        return {
            "ds": ds,
            "min_valid": min_valid,
            "normalize": normalize,
            "kept_class_indices": kept_indices,
            "per_class": [
                {"class": n, "n_valid": nv, "cm": cm.tolist() if not normalize else cm.tolist()}
                for (n, cm, nv) in per_class
            ],
            "save_path": save_path,
        }
    
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        monitor_loss,
        epoch,
        save_best=False,
        monitor_name="train_loss",
    ):
        def _atomic_torch_save(payload, destination):
            destination = os.fspath(destination)
            tmp_path = destination + ".tmp"
            os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
            try:
                torch.save(payload, tmp_path)
                os.replace(tmp_path, destination)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

        saved_best = False
        if save_best and monitor_loss is not None:
            if float(monitor_loss) < self.best_val_loss:
                print(
                    f"  [+] New Best Model "
                    f"({monitor_name}: {float(monitor_loss):.4f} < {self.best_val_loss:.4f})"
                )
                self.best_val_loss = float(monitor_loss)
                _atomic_torch_save(model.state_dict(), self.best_model_path)
                saved_best = True

        # Save the updated best scores, not their values from the previous epoch.
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'monitor_loss': float(monitor_loss) if monitor_loss is not None else None,
            'monitor_name': monitor_name,
            'best_val_loss': self.best_val_loss,
            'best_ultrasound_score': self.best_ultrasound_score,
        }
        last_path = os.path.join(self.ckpt_dir, "last_model.pth")
        _atomic_torch_save(state, last_path)
        return saved_best
            
    # def save_checkpoint(self, model, optimizer, scheduler, scaler, val_loss, epoch):
    #     # Create a comprehensive state dictionary
    #     state = {
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    #         'scaler_state_dict': scaler.state_dict() if scaler else None,
    #         'val_loss': val_loss,
    #         'best_val_loss': self.best_val_loss
    #     }

    #     # Save "Latest" (for resuming)
    #     last_path = os.path.join(self.ckpt_dir, "last_model.pth")
    #     torch.save(state, last_path)

    #     # Save "Best" (only if improved)
    #     if val_loss < self.best_val_loss:
    #         print(f"  [+] New Best Model (Val Loss: {val_loss:.4f} < {self.best_val_loss:.4f})")
    #         self.best_val_loss = val_loss
    #         # We usually just save weights for the best model to save space, 
    #         # but you can save the full state if you want.
    #         torch.save(model.state_dict(), self.best_model_path)

    # --- NEW: Plot ROC per Dataset ---
    def plot_dataset_roc_masked(self, y_true, y_probs, y_mask, class_names, ds_name, min_valid=5):
        """
        y_true:  [N, C]  multi-label targets (0/1)
        y_probs: [N, C]  predicted probabilities (0..1)
        y_mask:  [N, C]  1 if this class is valid for that sample's dataset, else 0
        """
        try:
            n_classes = len(class_names)
            plt.figure(figsize=(10, 8))

            auc_scores = []

            for i in range(n_classes):
                valid = (y_mask[:, i] == 1)

                # Not enough valid samples to estimate ROC meaningfully
                if valid.sum() < min_valid:
                    continue

                y_i = y_true[valid, i]
                p_i = y_probs[valid, i]

                # ROC needs both classes present
                if len(np.unique(y_i)) < 2:
                    continue

                fpr, tpr, _ = roc_curve(y_i, p_i)
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
                plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve (masked) - {ds_name}')
            plt.legend(loc="lower right", fontsize=9)
            plt.grid(alpha=0.3)

            clean_ds = ds_name.replace(" ", "_").replace("/", "-")
            plt.savefig(os.path.join(self.run_dir, f"ROC_masked_{clean_ds}.png"))
            plt.close()

            return float(np.mean(auc_scores)) if auc_scores else 0.0

        except Exception as e:
            print(f"Error plotting masked ROC for {ds_name}: {e}")
            return 0.0
        
# -----------------------------------------------------------------------------
# 3. Unified Datasets
# -----------------------------------------------------------------------------
# ===================== BUS-BRA descriptor constants (added) =====================
MEASUREMENT_KEYS = list(MEASUREMENT_KEYS_BUSBRA)
N_MEAS = int(N_MEAS_BUSBRA)
assert N_MEAS == 11, f"Expected 11 BUS-BRA measurements, got {N_MEAS}"
REPORT_MORPHOLOGY_KEYS = list(REPORT_MORPHOLOGY_KEYS_BUSBRA)


def compute_mask_measurements(mask):
    """Morphology vector from a binary mask (GT or PREDICTED). Same definitions as the
    BUS-BRA preprocessing so option-(B) predicted-mask measurements at inference match
    GT measurements at training. Returns (vec[float32, N_MEAS], present_bool)."""
    import math
    zeros = np.zeros(N_MEAS, dtype=np.float32)
    try:
        import cv2
    except Exception:
        return zeros, False
    if mask is None:
        return zeros, False
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim == 3:
        m = m.reshape(m.shape[-2], m.shape[-1]) if m.shape[0] == 1 else m.max(0)
    if m.sum() == 0:
        return zeros, False
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return zeros, False
    cnt = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(np.float32)
    max_d = perp_d = 0.0
    if len(hull) >= 2:
        diff = hull[:, None, :] - hull[None, :, :]
        d2 = np.sum(diff ** 2, -1)
        i, j = np.unravel_index(np.argmax(d2), d2.shape)
        vec = hull[j] - hull[i]
        max_d = float(np.linalg.norm(vec))
        if max_d > 1e-6:
            u = vec / max_d
            v = np.array([-u[1], u[0]], np.float32)
            proj = hull @ v
            perp_d = float(proj.max() - proj.min())
    area = float(m.sum())
    c_area = float(cv2.contourArea(cnt))
    perim = float(cv2.arcLength(cnt, True))
    x, y, w, h = cv2.boundingRect(cnt)
    bw, bh = float(w), float(h)
    aspect = bh / max(bw, 1.0)
    extent = area / max(float(w * h), 1.0)
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    solidity = c_area / max(hull_area, 1.0)
    circ = (4.0 * math.pi * c_area) / max(perim * perim, 1.0)
    ecc = 0.0
    if len(cnt) >= 5:
        (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
        maj, mnr = max(a1, a2), min(a1, a2)
        ecc = math.sqrt(max(0.0, 1.0 - (mnr ** 2) / max(maj ** 2, 1.0)))
    vals = {"max_diameter_px": max_d, "perpendicular_diameter_px": perp_d,
            "bbox_width_px": bw, "bbox_height_px": bh, "aspect_ratio_h_over_w": aspect,
            "area_px": area, "perimeter_px": perim, "circularity": circ,
            "solidity": solidity, "extent": extent, "eccentricity": ecc}
    return np.array([vals[k] for k in MEASUREMENT_KEYS], np.float32), True


def birads_ordinal_loss(logits, ordinal_target, mask=None, sample_weight=None):
    """BCE over cumulative thresholds; ordinal_target = [1]*c + [0]*(K-1-c) (matches JSON).

    sample_weight: optional [B] tensor (e.g. inverse-class-frequency weight
    per sample, looked up from its true BI-RADS category) to counter class
    imbalance -- without it, plain BCE lets the majority category dominate
    the gradient and starve minority categories (esp. the smallest one).
    """
    loss = F.binary_cross_entropy_with_logits(
        logits, ordinal_target.float(), reduction="none").sum(-1)
    if sample_weight is not None:
        loss = loss * sample_weight
    if mask is not None:
        m = mask.float()
        return (loss * m).sum() / m.sum().clamp_min(1.0)
    return loss.mean()


def birads_decode(logits, thresholds=None):
    """Decode cumulative CORAL logits with optional validation calibration."""
    q = torch.sigmoid(logits.float())
    if thresholds is None:
        thresholds = torch.full(
            (q.shape[-1],), 0.5, device=q.device, dtype=q.dtype
        )
    else:
        thresholds = torch.as_tensor(
            thresholds, device=q.device, dtype=q.dtype
        ).view(1, -1)
    return (q > thresholds).long().sum(-1)


def fit_birads_decode_thresholds(cumulative_probs, y_true, init=None):
    """Deterministic validation calibration for class-weighted CORAL logits."""
    q = np.asarray(cumulative_probs, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    if q.ndim != 2 or q.shape[0] != y.shape[0] or q.shape[0] < 8:
        return np.full(q.shape[1] if q.ndim == 2 else 3, 0.5, dtype=np.float32)
    t = np.asarray(
        init if init is not None else np.full(q.shape[1], 0.5),
        dtype=np.float64,
    ).copy()

    def score(thr):
        pred = (q > thr.reshape(1, -1)).sum(axis=1)
        acc = accuracy_score(y, pred)
        bal = balanced_accuracy_score(y, pred)
        mf1 = f1_score(y, pred, average='macro', zero_division=0)
        qwk = cohen_kappa_score(y, pred, weights='quadratic')
        if not np.isfinite(qwk):
            qwk = 0.0
        shrink = 0.01 * float(np.mean((thr - 0.5) ** 2))
        return 0.30 * acc + 0.20 * bal + 0.25 * mf1 + 0.25 * qwk - shrink

    for coarse in (True, False):
        for j in range(q.shape[1]):
            if coarse:
                candidates = np.arange(0.25, 0.751, 0.025)
            else:
                candidates = np.arange(
                    max(0.15, t[j] - 0.06),
                    min(0.85, t[j] + 0.0601),
                    0.01,
                )
            best_v, best_s = t[j], score(t)
            for v in candidates:
                cand = t.copy()
                cand[j] = float(v)
                sc = score(cand)
                if sc > best_s + 1e-12:
                    best_v, best_s = float(v), sc
            t[j] = best_v
    return np.clip(t, 0.15, 0.85).astype(np.float32)


def _model_birads_thresholds(model, device=None):
    core = model.module if hasattr(model, 'module') else model
    thresholds = getattr(core, 'birads_decode_thresholds', None)
    if thresholds is None:
        return None
    return thresholds.to(device=device) if device is not None else thresholds


def birads_class_probabilities(logits):
    """Convert K-1 cumulative CORAL probabilities into K class probabilities.

    For q_j=P(Y>j): p0=1-q0, pj=q[j-1]-q[j], p[K-1]=q[K-2].
    Small negative values caused by imperfect threshold ordering are clamped
    and rows are renormalized before multiclass ROC-AUC calculation.
    """
    q = torch.sigmoid(logits.float())
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError(f"Expected BI-RADS logits [B,K-1], got {tuple(q.shape)}")
    parts = [1.0 - q[:, :1]]
    if q.shape[1] > 1:
        parts.append(q[:, :-1] - q[:, 1:])
    parts.append(q[:, -1:])
    probs = torch.cat(parts, dim=1).clamp_min(0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)


def compute_birads_metrics(y_true, y_pred, y_prob=None, class_names=None, prefix=""):
    """Paper-standard multiclass + ordinal BI-RADS metrics."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.size == 0:
        return {}, None, []

    num_classes = int(max(y_true.max(), y_pred.max()) + 1)
    if class_names is not None:
        num_classes = max(num_classes, len(class_names))
    labels = np.arange(num_classes, dtype=np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out = {
        f"{prefix}Acc": float(accuracy_score(y_true, y_pred)),
        f"{prefix}BalancedAcc": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}MacroPrecision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}MacroRecall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}MacroF1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}WeightedF1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}MAE": float(np.mean(np.abs(y_pred - y_true))),
        f"{prefix}Within1": float(np.mean(np.abs(y_pred - y_true) <= 1)),
        f"{prefix}QWK": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.ndim == 2 and y_prob.shape[0] == y_true.shape[0]:
            if y_prob.shape[1] < num_classes:
                pad = np.zeros((y_prob.shape[0], num_classes - y_prob.shape[1]), dtype=y_prob.dtype)
                y_prob = np.concatenate([y_prob, pad], axis=1)
            elif y_prob.shape[1] > num_classes:
                y_prob = y_prob[:, :num_classes]
            y_prob = np.clip(y_prob, 0.0, None)
            y_prob = y_prob / np.clip(y_prob.sum(axis=1, keepdims=True), 1e-12, None)
            try:
                y_bin = label_binarize(y_true, classes=labels)
                out[f"{prefix}MacroAUC_OvR"] = float(
                    roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr")
                )
            except ValueError:
                out[f"{prefix}MacroAUC_OvR"] = float("nan")

    per_class = []
    for i in labels:
        name = class_names[i] if class_names is not None and i < len(class_names) else str(i)
        safe_name = str(name).replace(" ", "_")
        out[f"{prefix}Class{safe_name}_Precision"] = float(precision[i])
        out[f"{prefix}Class{safe_name}_Recall"] = float(recall[i])
        out[f"{prefix}Class{safe_name}_F1"] = float(f1[i])
        out[f"{prefix}Class{safe_name}_Support"] = int(support[i])
        per_class.append({
            "index": int(i), "name": str(name), "precision": float(precision[i]),
            "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i]),
        })

    return out, cm, per_class


def _collect_busbra_label_counts(dataset, sample_indices=None, num_birads_classes=4):
    """Count BUS-BRA labels either in the raw dataset or in a sampled index stream.

    ``sample_indices`` may contain duplicates. This is intentional: when the
    fixed-composition sampler oversamples a category, the count should reflect
    the frequency with which the loss actually sees that category.
    """
    birads_counts = np.zeros(int(num_birads_classes), dtype=np.float64)
    pathology_counts = np.zeros(2, dtype=np.float64)

    if sample_indices is None:
        sample_indices = range(len(dataset.data))

    for idx in sample_indices:
        s = dataset.data[int(idx)]
        if str(s.get("ds_name", "")) != "BUS_BRA":
            continue

        _b = s.get("birads_index", None)
        _hb = s.get("has_birads", True)
        if torch.is_tensor(_hb):
            _hb = bool(_hb.item())
        if _b is not None and _hb:
            _b = int(_b.item()) if torch.is_tensor(_b) else int(_b)
            if 0 <= _b < len(birads_counts):
                birads_counts[_b] += 1.0

        _p = s.get("pathology_index", None)
        _hp = s.get("has_pathology", True)
        if torch.is_tensor(_hp):
            _hp = bool(_hp.item())
        if _p is not None and _hp:
            _p = int(_p.item()) if torch.is_tensor(_p) else int(_p)
            if 0 <= _p < len(pathology_counts):
                pathology_counts[_p] += 1.0

    return birads_counts, pathology_counts


def estimate_busbra_training_stream_counts(dataset, batch_sampler=None, num_birads_classes=4):
    """Estimate the label distribution seen by the loss in one training epoch.

    For MoFX-Dense training, this follows the actual fixed-composition
    sampler. This avoids applying raw-dataset inverse-frequency weights on top
    of an already class-balanced sampler, which would double-correct class
    imbalance. In specialist/ordinary DataLoader mode, raw dataset counts are
    used because no custom batch sampler is available.
    """
    raw_b, raw_p = _collect_busbra_label_counts(
        dataset, sample_indices=None, num_birads_classes=num_birads_classes
    )

    if batch_sampler is None:
        sampled_b, sampled_p = raw_b.copy(), raw_p.copy()
        source = "raw_dataset"
    else:
        sampled_indices = []
        # __iter__ resets its work pools. The normal epoch loop calls
        # set_epoch(epoch) before DataLoader iteration, so this dry pass does
        # not alter the training sequence.
        for batch_indices in iter(batch_sampler):
            sampled_indices.extend(int(i) for i in batch_indices)
        sampled_b, sampled_p = _collect_busbra_label_counts(
            dataset,
            sample_indices=sampled_indices,
            num_birads_classes=num_birads_classes,
        )
        source = f"sampled_epoch({len(sampled_indices)} total records)"

    print(f"[BUS-BRA weighting] raw BI-RADS counts      = {raw_b.astype(int).tolist()}")
    print(f"[BUS-BRA weighting] sampled BI-RADS counts  = {sampled_b.astype(int).tolist()}")
    print(f"[BUS-BRA weighting] raw pathology counts    = {raw_p.astype(int).tolist()}")
    print(f"[BUS-BRA weighting] sampled pathology counts= {sampled_p.astype(int).tolist()}")
    print(f"[BUS-BRA weighting] source={source}")

    return {
        "raw_birads": raw_b,
        "raw_pathology": raw_p,
        "sampled_birads": sampled_b,
        "sampled_pathology": sampled_p,
        "source": source,
    }


def _tempered_balanced_weights(
    counts,
    power=0.5,
    min_weight=0.5,
    max_weight=3.0,
    extra_multipliers=None,
):
    """Return stable class-balanced weights with sample-weighted mean 1.

    Full inverse-frequency weighting is often too aggressive when a sampler
    already oversamples rare classes. Raising balanced weights to ``power``
    (default square root) preserves minority emphasis without double-counting
    the sampler. Clipping protects small classes from dominating one batch.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size == 0 or counts.sum() <= 0:
        return np.ones_like(counts, dtype=np.float64)

    present = counts > 0
    safe = np.where(present, counts, 1.0)
    raw = counts.sum() / (max(int(present.sum()), 1) * safe)
    weights = np.power(raw, float(power))
    weights[~present] = 1.0

    if extra_multipliers is not None:
        extra = np.asarray(extra_multipliers, dtype=np.float64)
        if extra.shape != weights.shape:
            raise ValueError(
                f"extra_multipliers shape {extra.shape} does not match weights {weights.shape}"
            )
        weights *= extra

    # Normalize using the distribution actually seen during training.
    mean_w = float((counts * weights).sum() / max(counts.sum(), 1.0))
    weights /= max(mean_w, 1e-12)
    weights = np.clip(weights, float(min_weight), float(max_weight))

    # Re-normalize after clipping so loss magnitude remains comparable.
    mean_w = float((counts * weights).sum() / max(counts.sum(), 1.0))
    weights /= max(mean_w, 1e-12)
    return weights


def compute_birads_class_weights(
    dataset,
    num_classes=4,
    counts=None,
    power=0.5,
    min_weight=0.5,
    max_weight=3.0,
):
    """Tempered class weights for BI-RADS categories.

    ``counts`` should normally be the sampled-stream counts returned by
    :func:`estimate_busbra_training_stream_counts`. Raw dataset counts are used
    only as a fallback.
    """
    if counts is None:
        counts, _ = _collect_busbra_label_counts(
            dataset, num_birads_classes=num_classes
        )
    weights = _tempered_balanced_weights(
        counts,
        power=power,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    print(f"[BI-RADS] weighting counts = {np.asarray(counts, dtype=int).tolist()}")
    print(
        f"[BI-RADS] tempered class weights (power={power:.3f}) = "
        f"{np.round(weights, 4).tolist()}"
    )
    return torch.tensor(weights, dtype=torch.float32)


def compute_pathology_class_weights(
    dataset,
    num_classes=2,
    counts=None,
    power=0.5,
    malignant_boost=1.15,
    min_weight=0.5,
    max_weight=3.0,
):
    """Tempered benign/malignant weights aligned with the sampled stream.

    A small configurable malignant boost is applied *after* correcting for the
    actual sampler distribution. It is deliberately mild because the sampler
    already contains a pathology-balanced BUS-BRA slot.
    """
    if counts is None:
        _, counts = _collect_busbra_label_counts(dataset)
    extra = np.ones(int(num_classes), dtype=np.float64)
    if int(num_classes) == 2:
        extra[1] = float(malignant_boost)
    weights = _tempered_balanced_weights(
        counts,
        power=power,
        min_weight=min_weight,
        max_weight=max_weight,
        extra_multipliers=extra,
    )
    print(f"[Pathology] weighting counts (benign, malignant) = {np.asarray(counts, dtype=int).tolist()}")
    print(
        f"[Pathology] tempered class weights (power={power:.3f}, "
        f"malignant_boost={malignant_boost:.3f}) = {np.round(weights, 4).tolist()}"
    )
    return torch.tensor(weights, dtype=torch.float32)


def compute_birads_bias_init(
    dataset,
    num_classes=4,
    counts=None,
    class_weights=None,
):
    """Initialize CORAL thresholds from the effective weighted train stream.

    Aligning the initial thresholds with both the sampler and the loss weights
    prevents the bias initialization from encoding raw-data prevalence while
    optimization is targeting a different, rebalanced distribution.
    """
    if counts is None:
        counts, _ = _collect_busbra_label_counts(
            dataset, num_birads_classes=num_classes
        )
    counts = np.asarray(counts, dtype=np.float64)
    if class_weights is not None:
        if torch.is_tensor(class_weights):
            class_weights = class_weights.detach().cpu().numpy()
        effective_counts = counts * np.asarray(class_weights, dtype=np.float64)
    else:
        effective_counts = counts

    total = effective_counts.sum()
    if total <= 0:
        print("[BI-RADS] No labeled samples found; using zero CORAL bias init.")
        return torch.zeros(num_classes - 1, dtype=torch.float32)

    cum_gt = total - np.cumsum(effective_counts)
    p_gt = np.clip(cum_gt[:-1] / total, 1e-4, 1 - 1e-4)
    bias_init = np.log(p_gt / (1.0 - p_gt))
    print(f"[BI-RADS] effective weighted counts = {np.round(effective_counts, 3).tolist()}")
    print(f"[BI-RADS] CORAL bias init = {np.round(bias_init, 4).tolist()}")
    return torch.tensor(bias_init, dtype=torch.float32)


def compute_pathology_bias_init(counts, class_weights=None):
    """Softmax bias initialization matched to sampled and weighted prevalence."""
    counts = np.asarray(counts, dtype=np.float64)
    if class_weights is not None:
        if torch.is_tensor(class_weights):
            class_weights = class_weights.detach().cpu().numpy()
        counts = counts * np.asarray(class_weights, dtype=np.float64)

    if counts.size != 2 or counts.sum() <= 0:
        return torch.zeros(2, dtype=torch.float32)

    prior = np.clip(counts / counts.sum(), 1e-6, 1.0)
    bias = np.log(prior)
    bias = bias - bias.mean()  # softmax is invariant to a common offset
    print(f"[Pathology] effective prior = {np.round(prior, 4).tolist()}")
    print(f"[Pathology] classifier bias init = {np.round(bias, 4).tolist()}")
    return torch.tensor(bias, dtype=torch.float32)


class BUSBRAJointSegTransform:
    """Conservative full-field BUS-BRA image/mask augmentation.

    The main model input always remains the complete ultrasound field of view.
    A lesion crop is generated later inside ``SwinMultiTask.forward`` from the
    GT/predicted lesion mask and is passed through the shared Swin encoder as a
    second visual view.  Consequently, this transform must never replace the
    original image with a crop.

    Only morphology-preserving augmentations are retained:
      * horizontal flip shared by image and mask;
      * mild image-only gamma and contrast changes.

    Rotation, translation, random scaling, Gaussian blur, speckle injection,
    random resized crop, and lesion-crop replacement are intentionally removed.
    """
    def __init__(
        self,
        target_size=(224, 224),
        train=True,
        hflip_p=0.50,
        gamma_p=0.20,
        gamma_range=(0.90, 1.10),
        contrast_p=0.20,
        contrast_range=(0.90, 1.10),
    ):
        self.size_hw = (target_size[1], target_size[0])
        self.train = bool(train)
        self.hflip_p = float(hflip_p)
        self.gamma_p = float(gamma_p)
        self.gamma_range = tuple(float(v) for v in gamma_range)
        self.contrast_p = float(contrast_p)
        self.contrast_range = tuple(float(v) for v in contrast_range)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, image, mask):
        # Keep the full field of view for the primary image stream.
        image = TF.resize(
            image,
            self.size_hw,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            self.size_hw,
            interpolation=InterpolationMode.NEAREST,
        )

        # Horizontal reflection preserves lesion morphology and is shared by
        # image and mask so segmentation and ROI extraction remain aligned.
        if self.train and random.random() < self.hflip_p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = (mask > 0.5).float()

        # Mild scanner/intensity variation only.  No blur or synthetic speckle:
        # both can erase margin/echo-pattern evidence needed for BI-RADS and
        # pathology classification.
        if self.train and random.random() < self.gamma_p:
            image = TF.adjust_gamma(
                image,
                random.uniform(*self.gamma_range),
            )
        if self.train and random.random() < self.contrast_p:
            image = TF.adjust_contrast(
                image,
                random.uniform(*self.contrast_range),
            )

        image = image.clamp(0.0, 1.0)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, mask

class XRayJointSegTransform:
    """Apply the paper's X-ray geometry identically to image and mask."""

    def __init__(
        self,
        target_size=(224, 224),
        train=True,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.target_size = tuple(target_size)
        self.train = bool(train)
        self.mean = tuple(mean)
        self.std = tuple(std)

    def __call__(self, image, mask):
        output_size = (self.target_size[1], self.target_size[0])
        image = TF.resize(
            image,
            output_size,
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            output_size,
            interpolation=InterpolationMode.NEAREST,
        )

        if self.train and random.random() < 0.7:
            angle, translate, scale, shear = transforms.RandomAffine.get_params(
                degrees=(-8.0, 8.0),
                translate=(0.05, 0.05),
                scale_ranges=(0.92, 1.08),
                shears=None,
                img_size=[self.target_size[0], self.target_size[1]],
            )
            affine_args = {
                "angle": angle,
                "translate": translate,
                "scale": scale,
                "shear": shear,
                "fill": 0,
            }
            image = TF.affine(
                image,
                interpolation=InterpolationMode.BICUBIC,
                **affine_args,
            )
            mask = TF.affine(
                mask,
                interpolation=InterpolationMode.NEAREST,
                **affine_args,
            )

        if self.train:
            top, left, height, width = transforms.RandomResizedCrop.get_params(
                image,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            )
            crop_args = (top, left, height, width, output_size)
            image = TF.resized_crop(
                image,
                *crop_args,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask,
                *crop_args,
                interpolation=InterpolationMode.NEAREST,
            )
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        mask = (mask > 0.5).float()
        return image, mask


class UnifiedMedicalDataset(Dataset):
    def __init__(
        self,
        data_root_folder,
        split='train',
        transform_dict=None,
        tokenizer=None,
        target_size=(224, 224),
        max_seq_len=512,
        cache_image_sizes=True,
        cache_images_in_ram=False,
        cache_targets_in_ram=True,
        use_report_label_guidance=True,
    ):
        self.transform_dict = transform_dict or {}
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_image_sizes = cache_image_sizes
        self.use_report_label_guidance = bool(use_report_label_guidance)

        self.data = []
        self.full_slice_mri_datasets = {"LSS_MRI_AISSLab"}
        self.mri_roi_crop_cls_datasets = {"LSS_MRI_ROI_CLS"}
        self.NORMAL_LABEL = "No_Finding"

        # RAM caches
        self.cache_images_in_ram = cache_images_in_ram
        self.image_cache = {} if cache_images_in_ram else None

        self.cache_targets_in_ram = cache_targets_in_ram
        self.seg_cache = {} if cache_targets_in_ram else None
        self.det_cache = {} if cache_targets_in_ram else None
        
        self.multiclass_cls_datasets = {
            "LSS_MRI_ROI_CLS",
            # later add your CT / US dataset names here if they are also exclusive 3-class problems
        }
        self.split = split

        self.busbra_joint_transform = BUSBRAJointSegTransform(
            target_size=target_size,
            train=(split == "train"),
        )
        self.xray_joint_seg_transform = XRayJointSegTransform(
            target_size=target_size,
            train=(split == "train"),
        )

        # ---------------------------------------------------------
        # 1. Load config
        # ---------------------------------------------------------
        config = {'datasets': {}}
        json_files = glob.glob(os.path.join(data_root_folder, "*.json"))
        for jf in json_files:
            with open(jf, 'r') as f:
                d = json.load(f)
                if 'datasets' in d:
                    config['datasets'].update(d['datasets'])
        self.config = config

        # ---------------------------------------------------------
        # 2. Build modality map
        # ---------------------------------------------------------
        unique_modalities = set()
        self.ds_to_modality_name = {}
        for ds_name, ds_config in config['datasets'].items():
            mod_name = ds_config.get('modality', ds_name)
            unique_modalities.add(mod_name)
            self.ds_to_modality_name[ds_name] = mod_name

        self.modality_names = sorted(list(unique_modalities))
        self.modality_map = {name: i for i, name in enumerate(self.modality_names)}
        print(f"Modality Map: {self.modality_map}")

        # ---------------------------------------------------------
        # 3. Build classification schema
        # ---------------------------------------------------------
        unique_classes = set()
        for _ds_name, ds_config in config['datasets'].items():
            # BUS-BRA uses dedicated BI-RADS and pathology heads.  Do not add
            # its duplicate benign/malignant labels to the generic schema.
            if _ds_name == "BUS_BRA":
                continue
            if 'classification' in ds_config.get('tasks', []):
                ds_classes = ds_config.get('classes', ds_config.get('class_names', []))

                if not ds_classes and ds_config.get('type') == 'json_dataset':
                    recs = ds_config.get('rec', [])
                    if recs:
                        raw_keys = list(
                            recs[0].get('annotations', {})
                                 .get('classification_labels', {})
                                 .keys()
                        )
                        ds_classes = [k.replace("No_Finding", self.NORMAL_LABEL) for k in raw_keys]

                if ds_classes:
                    unique_classes.update(ds_classes)

        if not unique_classes:
            unique_classes = {self.NORMAL_LABEL}

        self.class_names = sorted(list(unique_classes))
        self.class_map = {name: i for i, name in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)
        print(f"Global Classification Schema ({self.num_classes} classes): {self.class_names}")

        # ---------------------------------------------------------
        # 4. Build segmentation schema
        # ---------------------------------------------------------
        seg_classes_set = set()
        for ds_config in config['datasets'].values():
            if 'segmentation' in ds_config.get('tasks', []):
                recs = ds_config.get('rec', [])
                if recs:
                    seg_def = recs[0].get('annotations', {}).get('segmentation_classes', {})
                    for v in seg_def.values():
                        if v.lower() != "background":
                            seg_classes_set.add(v)

        self.seg_class_names = sorted(list(seg_classes_set))
        self.seg_class_map = {name: i for i, name in enumerate(self.seg_class_names)}
        self.num_seg_classes = len(self.seg_class_names)
        print(f"Global Segmentation Schema ({self.num_seg_classes} classes): {self.seg_class_names}")

        # ---------------------------------------------------------
        # 5. Build detection schema
        # ---------------------------------------------------------
        det_classes_set = set()
        for ds_config in config['datasets'].values():
            if 'detection' in ds_config.get('tasks', []):
                recs = ds_config.get('rec', [])
                for rec in recs[-50:]:
                    det = rec.get('annotations', {}).get('detection_targets', {})
                    label_map = det.get('label_map', {})
                    for v in label_map.values():
                        det_classes_set.add(v)

        if len(det_classes_set) == 0:
            det_classes_set = {"object"}

        self.det_class_names = sorted(list(det_classes_set))
        self.det_class_map = {name: i for i, name in enumerate(self.det_class_names)}

        # ---- BI-RADS schema (BUS-BRA only; separate gated head) ----
        self.birads_class_names = []
        for ds_config in config['datasets'].values():
            if 'birads_classification' in ds_config.get('tasks', []):
                self.birads_class_names = [str(c) for c in ds_config.get('birads_class_names', [])]
                break
        self.birads_num_classes = len(self.birads_class_names) if self.birads_class_names else 4
        print(f"Global Detection Schema ({len(self.det_class_names)} classes): {self.det_class_names}")

        # ---------------------------------------------------------
        # 6. Build flattened sample list with precomputed metadata
        # ---------------------------------------------------------
        for ds_name, ds_config in config['datasets'].items():
            ds_type = ds_config.get('type', 'json_dataset')
            default_tasks = ds_config.get('tasks', [])

            if ds_type != 'json_dataset':
                continue

            mod_name = self.ds_to_modality_name[ds_name]
            mod_id = self.modality_map[mod_name]

            for rec in ds_config.get('rec', []):
                if rec.get('split') != split:
                    continue

                img_path = rec['inputs']['image_path']
                if not os.path.exists(img_path):
                    continue

                current_tasks = rec.get('tasks', default_tasks).copy()
                is_full_slice_mri = ds_name in self.full_slice_mri_datasets
                is_mri_roi_crop_cls = ds_name in self.mri_roi_crop_cls_datasets

                if (not is_full_slice_mri) and 'detection' in current_tasks:
                    current_tasks.remove('detection')

                # only full-slice MRI should have generic classification removed
                if is_full_slice_mri and 'classification' in current_tasks:
                    current_tasks.remove('classification')

                annot = rec.get('annotations', {})

                if 'report_generation' in current_tasks:
                    report_text = annot.get('report_text') or annot.get('findings') or annot.get('impression')
                    if not report_text or len(str(report_text).strip()) < 5:
                        current_tasks.remove('report_generation')

                if len(current_tasks) == 0:
                    continue

                # ---------- precompute task flags once ----------
                has_seg = 'segmentation' in current_tasks
                has_cls_task = 'classification' in current_tasks
                # BUS-BRA has dedicated BI-RADS/pathology heads; its duplicate
                # benign/malignant generic target is intentionally disabled.
                has_generic_cls = has_cls_task and ds_name != "BUS_BRA"
                has_rep = 'report_generation' in current_tasks or any('report' in t for t in current_tasks)
                has_det = 'detection' in current_tasks

                sample = {
                    'img_path': img_path,
                    'tasks': current_tasks,
                    'ds_name': ds_name,
                    'mod_name': mod_name,
                    'modality_id': torch.tensor(mod_id, dtype=torch.long),
                    'target_class': "None",
                    'record': rec,

                    'has_seg': torch.tensor(has_seg, dtype=torch.bool),
                    'has_cls': torch.tensor(has_generic_cls, dtype=torch.bool),
                    'has_rep': torch.tensor(has_rep, dtype=torch.bool),
                    'has_det': torch.tensor(has_det, dtype=torch.bool),
                    'task_mask': torch.tensor(
                        [float(has_seg), float(has_cls_task), float(has_rep), float(has_det)],
                        dtype=torch.float32
                    ),
                    'is_lss_mri': torch.tensor(is_full_slice_mri, dtype=torch.bool),
                }

                # ---- BUS-BRA: BI-RADS + measurements + pathology ----
                # SOURCE OF TRUTH = the annotation, NOT the record's `tasks` list.
                # Descriptor GT must be present whenever the annotation carries it
                # (e.g. a report-only record still has measurements/BI-RADS/pathology),
                # so specialist report mode can condition on GT at val/test.
                # `has_birads` (the supervised-loss gate) still follows the task list.
                has_birads = (
                    'birads_classification' in current_tasks
                    or (ds_name == "BUS_BRA" and has_cls_task)
                )
                sample['has_birads'] = torch.tensor(has_birads, dtype=torch.bool)

                # BI-RADS index: read from annotation if available (ungated)
                _bir = annot.get('birads_classification', {}) or {}
                _bidx = int(_bir.get('class_index', -100)) if _bir else -100
                sample['birads_index'] = torch.tensor(_bidx, dtype=torch.long)
                sample['has_birads'] = torch.tensor(
                    bool(has_birads and _bidx >= 0), dtype=torch.bool
                )
                _K = self.birads_num_classes
                _ord = _bir.get('ordinal_target', None)
                if _ord is not None:
                    _ord_t = torch.tensor([float(v) for v in _ord], dtype=torch.float32)
                else:
                    _ord_t = torch.zeros(max(_K - 1, 1), dtype=torch.float32)
                    if _bidx >= 0:
                        _ord_t[:_bidx] = 1.0
                sample['birads_ordinal'] = _ord_t

                # Measurements: read from annotation (ungated)
                _meas = annot.get('measurements', {}) or {}
                sample['measurements'] = torch.tensor(
                    [float(_meas.get(k, 0.0)) for k in MEASUREMENT_KEYS], dtype=torch.float32)
                _mpres = bool(_meas) and any(float(_meas.get(k, 0.0)) != 0.0 for k in MEASUREMENT_KEYS)
                sample['measurements_present'] = torch.tensor(_mpres, dtype=torch.bool)

                # Preserve the complete GT report morphology (including fitted-ellipse
                # axes) for source-space audit/comparison. This does not enter the
                # learned 11-D MeasurementEncoder and therefore does not change the
                # trained checkpoint interface.
                if ds_name == "BUS_BRA":
                    _ratio = _meas.get(
                        'major_minor_ratio',
                        _meas.get('axis_ratio_major_minor', 0.0),
                    )
                    sample['report_morphology_gt'] = {
                        'max_diameter_px': float(_meas.get('max_diameter_px', 0.0)),
                        'perpendicular_diameter_px': float(_meas.get('perpendicular_diameter_px', 0.0)),
                        'bbox_width_px': float(_meas.get('bbox_width_px', 0.0)),
                        'bbox_height_px': float(_meas.get('bbox_height_px', 0.0)),
                        'aspect_ratio_h_over_w': float(_meas.get('aspect_ratio_h_over_w', 0.0)),
                        'area_px': float(_meas.get('area_px', 0.0)),
                        'perimeter_px': float(_meas.get('perimeter_px', 0.0)),
                        'circularity': float(_meas.get('circularity', 0.0)),
                        'solidity': float(_meas.get('solidity', 0.0)),
                        'extent': float(_meas.get('extent', 0.0)),
                        'eccentricity': float(_meas.get('eccentricity', 0.0)),
                        'major_axis_px': float(_meas.get('major_axis_px', 0.0)),
                        'minor_axis_px': float(_meas.get('minor_axis_px', 0.0)),
                        'major_minor_ratio': float(_ratio or 0.0),
                    }

                    # Explicit BUS-BRA lesion channel(s) in the unified segmentation
                    # schema. Needed so report morphology never assumes channel 0.
                    _seg_def = annot.get('segmentation_classes', {}) or {}
                    _bus_ch = []
                    for _class_name in _seg_def.values():
                        _class_name = str(_class_name).strip()
                        if (not _class_name) or ('background' in _class_name.lower()):
                            continue
                        if _class_name in self.seg_class_map:
                            _bus_ch.append(int(self.seg_class_map[_class_name]))
                    sample['busbra_seg_channel_indices'] = tuple(sorted(set(_bus_ch)))

                # BUS-BRA GT report measurements were computed from the source
                # segmentation mask.  Preserve that mask size so predicted masks
                # can be mapped back to the same pixel coordinate system at
                # validation/test time.
                if ds_name == "BUS_BRA":
                    _measurement_mask_path = annot.get('segmentation_mask')
                    try:
                        if _measurement_mask_path and os.path.exists(_measurement_mask_path):
                            with Image.open(_measurement_mask_path) as _mim:
                                _mw, _mh = _mim.size
                        else:
                            with Image.open(img_path) as _iim:
                                _mw, _mh = _iim.size
                    except Exception:
                        _mw, _mh = self.target_size
                    sample['measurement_size'] = (int(_mw), int(_mh))

                # Pathology index: read from annotation (ungated)
                _pc = annot.get('pathology_classification', {}) or {}
                _pidx = int(_pc.get('class_index', -100)) if _pc else -100
                sample['pathology_index'] = torch.tensor(_pidx, dtype=torch.long)
                sample['has_pathology'] = torch.tensor(
                    bool(ds_name == "BUS_BRA" and has_cls_task and _pidx >= 0),
                    dtype=torch.bool,
                )

                # Cache original image size once
                if self.cache_image_sizes:
                    try:
                        with Image.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        w, h = self.target_size
                    sample['orig_size'] = (w, h)

                # -------------------------------------------------
                # Precompute classification tensors
                # -------------------------------------------------
                if has_generic_cls:
                    if ds_name in self.multiclass_cls_datasets:
                        cls_labels, cls_mask, active_tags, class_index, local_class_indices = \
                            self._build_roi_crop_classification_targets(annot)

                        sample['cls_problem_type'] = torch.tensor(1, dtype=torch.long)   # 1 = multiclass
                        sample['class_index'] = torch.tensor(class_index, dtype=torch.long)
                        sample['local_class_indices'] = torch.tensor(local_class_indices, dtype=torch.long)
                    else:
                        cls_labels, cls_mask, active_tags = self._build_classification_targets(annot)

                        sample['cls_problem_type'] = torch.tensor(0, dtype=torch.long)   # 0 = multilabel
                        sample['class_index'] = torch.tensor(-100, dtype=torch.long)
                        sample['local_class_indices'] = torch.nonzero(cls_mask > 0, as_tuple=False).view(-1).long()

                    sample['class_labels'] = cls_labels
                    sample['class_mask'] = cls_mask
                    sample['active_tags'] = active_tags
                else:
                    sample['active_tags'] = []

                if is_full_slice_mri:
                    sample['mri_active_tags'] = self._build_mri_active_tags(annot)
                else:
                    sample['mri_active_tags'] = []

                # -------------------------------------------------
                # Precompute segmentation metadata
                # -------------------------------------------------
                if has_seg:
                    sample['seg_mask_path'] = annot.get('segmentation_mask')
                    sample['seg_local_classes'] = annot.get('segmentation_classes', {})
                    sample['target_class'] = "Multi-Anatomy"

                # -------------------------------------------------
                # Precompute detection metadata
                # -------------------------------------------------
                if has_det and is_full_slice_mri:
                    det = annot.get('detection_targets', {})
                    sample['det_boxes_raw'] = det.get('boxes', [])
                    sample['det_labels_raw'] = det.get('labels', [])
                    sample['det_severity_raw'] = det.get('severity', [])
                    sample['det_label_map'] = det.get('label_map', {})

                # -------------------------------------------------
                # Pre-tokenize report once
                # -------------------------------------------------
                if has_rep and self.tokenizer is not None:
                    report_text = annot.get('report_text') or annot.get('findings') or annot.get('impression') or ""

                    if is_full_slice_mri:
                        report_ids, report_labels = self._build_report_tokens_mri(
                            mod_name=mod_name,
                            active_tags=sample['mri_active_tags'],
                            target_report=report_text,
                            use_tag_prompt=self.use_report_label_guidance,
                        )
                    else:
                        # BUS-BRA: descriptor-aware prompt from GT (train teacher-forcing)
                        _rep_desc = None
                        _mm = annot.get('measurements', {}) or {}
                        _pc = annot.get('pathology_classification', {}) or {}
                        _bc = annot.get('birads_classification', {}) or {}
                        if _mm or _pc or _bc:
                            _rep_desc = {
                                'max_diameter_px': _mm.get('max_diameter_px'),
                                'perpendicular_diameter_px': _mm.get('perpendicular_diameter_px'),
                                'orientation': _mm.get('orientation_descriptor'),
                                'pathology': _pc.get('class_name'),
                                'birads': _bc.get('category'),
                            }
                        report_ids, report_labels = self._build_report_tokens(
                            mod_name=mod_name,
                            active_tags=sample['active_tags'],
                            target_report=report_text,
                            use_tag_prompt=self.use_report_label_guidance,
                            descriptors=_rep_desc,
                        )

                    sample['report_ids'] = report_ids
                    sample['report_labels'] = report_labels
                    sample['report_text'] = report_text
                else:
                    sample['report_text'] = ""

                self.data.append(sample)

        # ---------------------------------------------------------
        # 7. Build modality index list
        # ---------------------------------------------------------
        self.modality_to_indices = {i: [] for i in range(len(self.modality_map))}
        for idx, item in enumerate(self.data):
            self.modality_to_indices[item['modality_id'].item()].append(idx)

    def __len__(self):
        return len(self.data)

    def compute_measurement_stats(self):
        """Train-set mean/std over samples that HAVE measurements (present=True).
        Pass these to the model's MeasurementEncoder.set_stats so val/test use
        TRAIN normalization. Returns (mean[N_MEAS], std[N_MEAS]) or (None, None)."""
        vs = [s['measurements'] for s in self.data
              if bool(s.get('measurements_present', torch.tensor(False)))]
        if not vs:
            return None, None
        M = torch.stack(vs)
        return M.mean(0), M.std(0).clamp_min(1e-6)

    # Cross-task dependency map: for each (ds_name, primary_task), which other
    # tasks must EXECUTE their forward path so their outputs are available.
    # "Executed" != "supervised" — dependent tasks contribute no training loss,
    # but their predictions (masks, boxes) are consumed by the primary task.
    #
    # BUS-BRA/cls   needs seg: predicted mask -> measurements -> BI-RADS/pathology
    # BUS-BRA/rep   needs seg: seg-guided report tokens
    # CXLSeg/cls    needs seg: seg-guided classification attention
    # CXLSeg/rep    needs seg: seg-guided report tokens
    # LSS_MRI_*/cls needs det: predicted boxes -> MRI ROI classification
    # LSS_MRI_*/rep needs det: MRI ROI boxes -> roi_to_report tokens
    _CROSS_TASK_DEPS = {
        # (ds_name_prefix, primary_task): set of dependent task names to keep EXECUTED
        ("BUS_BRA",        "classification"):    {"segmentation"},
        ("BUS_BRA",        "report_generation"): {"segmentation"},
        ("BUS-BRA",        "classification"):    {"segmentation"},
        ("BUS-BRA",        "report_generation"): {"segmentation"},
        ("CXLSeg",         "classification"):    {"segmentation"},
        ("CXLSeg",         "report_generation"): {"segmentation"},
        ("LSS_MRI_ROI_CLS","classification"):    {"detection"},
        ("LSS_MRI_AISSLab","report_generation"): {"detection"},
    }

    def make_specialist(self, ds_name, task):
        """Restrict this split to a single (ds_name, task) specialist.

        PRIMARY task: supervised with full loss; has_<task>=True; task_mask[ti]=1.
        DEPENDENT tasks: their forward path executes (so predictions are available
          to the primary task) but has_*=False -> they contribute NO training loss.
          task_mask is set to 1 for them so the forward runs; has_* is False so
          the loss functions gate them out.
        UNRELATED tasks: task_mask=0 and has_*=False (fully suppressed).

        Examples
        --------
        BUS-BRA / classification:
          -> seg executes (mask -> measurements -> BI-RADS), not supervised.
        LSS_MRI_ROI_CLS / classification:
          -> det executes (boxes -> MRI ROI), not supervised.
        CXLSeg / segmentation:
          -> no dependencies; only seg executes.
        """
        TASK_IDX  = {"segmentation": 0, "classification": 1,
                     "report_generation": 2, "detection": 3}
        TASK_KEYS = {v: k for k, v in TASK_IDX.items()}
        if task not in TASK_IDX:
            raise ValueError("task must be one of %s" % list(TASK_IDX))
        ti = TASK_IDX[task]

        # Resolve which dependent tasks need their forward to execute
        dep_tasks = set()
        for (ds_prefix, ptask), deps in self._CROSS_TASK_DEPS.items():
            if ptask == task and (ds_name is None or
                                  (ds_name is not None and
                                   (ds_name == ds_prefix or ds_name.startswith(ds_prefix)))):
                dep_tasks |= deps
        dep_indices = {TASK_IDX[t] for t in dep_tasks if t in TASK_IDX}

        kept = []
        for s in self.data:
            if ds_name is not None and s["ds_name"] != ds_name:
                continue
            if float(s["task_mask"][ti].item()) <= 0.5:
                continue

            # Build task_mask: 1 for primary + dependent tasks, 0 for unrelated.
            # The dependent flag in task_mask=1 lets the forward path execute;
            # has_* stays False so the loss is gated out.
            tm = torch.zeros(4, dtype=torch.float32)
            tm[ti] = 1.0                              # primary supervised
            for dep_ti in dep_indices:
                tm[dep_ti] = 1.0                      # dependent: execute, not supervised
            s["task_mask"] = tm

            # has_* = True ONLY for the primary task (loss is gated on these)
            s["has_seg"] = torch.tensor(task == "segmentation", dtype=torch.bool)
            s["has_cls"] = torch.tensor(
                task == "classification" and s.get("ds_name") != "BUS_BRA",
                dtype=torch.bool,
            )
            s["has_rep"] = torch.tensor(task == "report_generation", dtype=torch.bool)
            s["has_det"] = torch.tensor(task == "detection", dtype=torch.bool)

            # BI-RADS: keep active when cls is primary (it IS the cls signal);
            # suppress otherwise.
            if "has_birads" in s:
                s["has_birads"] = torch.tensor(
                    task == "classification" and int(s.get("birads_index", -100)) >= 0,
                    dtype=torch.bool,
                )
            if "has_pathology" in s:
                _pidx = int(s.get("pathology_index", torch.tensor(-100)).item())
                s["has_pathology"] = torch.tensor(
                    task == "classification" and _pidx >= 0,
                    dtype=torch.bool,
                )

            kept.append(s)

        self.data = kept
        if hasattr(self, "modality_to_indices"):
            self.modality_to_indices = {i: [] for i in range(len(self.modality_map))}
            for idx, item in enumerate(self.data):
                self.modality_to_indices[item["modality_id"].item()].append(idx)

        n_birads = sum(1 for s in self.data
                       if bool(s.get("has_birads", torch.tensor(False))))
        dep_str  = (", ".join(sorted(dep_tasks)) if dep_tasks else "none")
        print("[specialist] %s / %s -> %d samples  "
              "(has_birads=%d, dependent-tasks-executing=%s)" % (
              ds_name, task, len(self.data), n_birads, dep_str))
        return len(self.data)

    def _get_transform(self, mod_name, ds_name=None):
        if ds_name is not None and ds_name in self.transform_dict:
            return self.transform_dict[ds_name]
        if mod_name in self.transform_dict:
            return self.transform_dict[mod_name]
        if 'default' in self.transform_dict:
            return self.transform_dict['default']
        return transforms.ToTensor()

    def _build_mri_active_tags(self, annot):
        """
        Build MRI report tags from GT detection labels + GT severity.
        Example output: ["L4-L5 Moderate", "L5-S1 Severe"]
        """
        det = annot.get('detection_targets', {}) or {}
        labels = det.get('labels', []) or []
        severity = det.get('severity', []) or []
        label_map = det.get('label_map', {}) or {}

        severity_names = {
            0: "Normal",
            1: "Mild",
            2: "Moderate",
            3: "Severe",
        }

        tags = []
        n = min(len(labels), len(severity))

        for i in range(n):
            try:
                level_id = int(labels[i])
                sev_id = int(severity[i])
            except Exception:
                continue

            # map label id -> vertebral level string
            level_name = (
                label_map.get(level_id, None)
                or label_map.get(str(level_id), None)
                or f"Level{level_id}"
            )

            sev_name = severity_names.get(sev_id, f"Severity{sev_id}")

            # usually skip "Normal" so prompt stays focused like non-MRI prompts
            if sev_id == 0:
                continue

            tags.append(f"{level_name} {sev_name}")

        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for t in tags:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def _build_report_tokens_mri(self, mod_name, active_tags, target_report, use_tag_prompt=True):
        if not use_tag_prompt:
            content_str = "Describe the MRI findings. Report:"
        elif active_tags:
            tag_str = ", ".join(active_tags)
            content_str = f"Detected abnormalities: {tag_str}. Report:"
        else:
            content_str = "No abnormality detected. Report:"

        messages = [
            {"role": "system", "content": f"You are an expert {mod_name} radiologist. Describe the findings."},
            {"role": "user", "content": content_str}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_text = prompt_text + " " + target_report + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        report_ids = self.tokenizer(target_report + self.tokenizer.eos_token, add_special_tokens=False)['input_ids']

        available = self.max_seq_len - len(prompt_ids)

        if available < 10:
            input_ids = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(0)
        else:
            final_ids = prompt_ids + report_ids[:available]
            if len(final_ids) < self.max_seq_len:
                final_ids = final_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(final_ids))
            else:
                final_ids = final_ids[:self.max_seq_len]
            input_ids = torch.tensor(final_ids, dtype=torch.long)

        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        eos_indices = (labels == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        report_eos_indices = eos_indices[eos_indices >= prompt_len]

        if len(report_eos_indices) > 0:
            stop_idx = report_eos_indices[0].item()
            if stop_idx + 1 < len(labels):
                labels[stop_idx + 1:] = -100
        else:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

    def _build_roi_crop_classification_targets(self, annot):
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        label_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        active_tags = []

        class_id = annot.get("classification_label", None)
        class_name = annot.get("classification_name", None)
        class_map_local = annot.get("classification_map", {}) or {}

        ordered_keys = sorted(class_map_local.keys(), key=lambda x: int(x))
        ordered_names = [str(class_map_local[k]).replace("No_Finding", self.NORMAL_LABEL) for k in ordered_keys]

        local_class_indices = []
        for name in ordered_names:
            if name in self.class_map:
                gidx = self.class_map[name]
                label_mask[gidx] = 1.0
                local_class_indices.append(gidx)

        resolved_name = None
        if class_name is not None:
            resolved_name = str(class_name).replace("No_Finding", self.NORMAL_LABEL)
        elif class_id is not None:
            resolved_name = (
                class_map_local.get(str(class_id), None)
                or class_map_local.get(int(class_id), None)
            )
            if resolved_name is not None:
                resolved_name = str(resolved_name).replace("No_Finding", self.NORMAL_LABEL)

        class_index = -100
        if resolved_name is not None and resolved_name in self.class_map:
            labels[self.class_map[resolved_name]] = 1.0
            if resolved_name != self.NORMAL_LABEL:
                active_tags.append(resolved_name)

            if resolved_name in ordered_names:
                class_index = ordered_names.index(resolved_name)

        return labels, label_mask, active_tags, class_index, local_class_indices

    def _build_classification_targets(self, annot):
        labels = torch.zeros(self.num_classes, dtype=torch.float32)
        label_mask = torch.zeros(self.num_classes, dtype=torch.float32)
        active_tags = []

        annot_labels = annot.get('classification_labels', {})
        for k, v in annot_labels.items():
            norm_k = k.replace("No_Finding", self.NORMAL_LABEL)
            if norm_k in self.class_map:
                idx = self.class_map[norm_k]
                label_mask[idx] = 1.0
                if v == 1:
                    labels[idx] = 1.0
                    if norm_k != self.NORMAL_LABEL:
                        active_tags.append(norm_k)

        return labels, label_mask, active_tags

    def _build_descriptor_content(self, d, enforce_pixel_units=False):
        """BUS-BRA descriptor-aware report prompt.

        Training remains backward compatible because old training descriptors contain
        only the fields they previously supplied. At validation/test time, the full
        source-space predicted morphology dictionary is provided, so every quantitative
        value appearing in the GT report is explicitly grounded in the predicted mask.
        """
        parts = ["Breast ultrasound."]

        # Clinical findings first.
        bir = d.get("birads")
        if bir is not None and str(bir).strip() != "":
            parts.append(f"BI-RADS: {bir}.")
        path = d.get("pathology")
        if path:
            parts.append(f"Pathology: {path}.")

        def _num(key):
            try:
                value = d.get(key, None)
                if value is None:
                    return None
                value = float(value)
                return value if np.isfinite(value) else None
            except (TypeError, ValueError):
                return None

        # All mask-derived values that occur in the GT BUS-BRA reports.
        md = _num("max_diameter_px")
        pd = _num("perpendicular_diameter_px")
        maj = _num("major_axis_px")
        mnr = _num("minor_axis_px")
        ratio = _num("major_minor_ratio")
        aspect = _num("aspect_ratio_h_over_w")
        area = _num("area_px")
        perim = _num("perimeter_px")
        circ = _num("circularity")
        solid = _num("solidity")
        extent = _num("extent")
        ecc = _num("eccentricity")

        q = []
        if md is not None and md > 0:
            q.append(f"maximum diameter {md:.1f} px")
        if pd is not None and pd > 0:
            q.append(f"perpendicular extent {pd:.1f} px")
        if maj is not None and mnr is not None and maj > 0 and mnr > 0:
            q.append(f"ellipse major/minor {maj:.1f}/{mnr:.1f} px")
        if ratio is not None and ratio > 0:
            q.append(f"major/minor ratio {ratio:.2f}")
        if aspect is not None and aspect >= 0:
            q.append(f"aspect H/W {aspect:.2f}")
        if area is not None and area >= 0:
            q.append(f"area {area:.0f} px^2")
        if perim is not None and perim >= 0:
            q.append(f"perimeter {perim:.1f} px")
        if circ is not None and circ >= 0:
            q.append(f"circularity {circ:.2f}")
        if solid is not None and solid >= 0:
            q.append(f"solidity {solid:.2f}")
        if extent is not None and extent >= 0:
            q.append(f"extent {extent:.2f}")
        if ecc is not None and ecc >= 0:
            q.append(f"eccentricity {ecc:.2f}")
        if q:
            parts.append("Measurements: " + "; ".join(q) + ".")

        orient = d.get("orientation")
        if orient:
            parts.append(f"Orientation: {orient}.")

        if enforce_pixel_units:
            parts.append(
                "Use the supplied quantitative values exactly. Keep pixel measurements "
                "in px only; do not convert, alter, or invent numeric values."
            )
        parts.append("Report:")
        return " ".join(parts)

    def build_gt_descriptor_dict(self, meas_vec, birads_idx, pathology_idx, full_morphology=None):
        """Assemble a descriptor dict from GT batch tensors (for specialist
        report mode, where predicted heads are untrained and must not be used).
        meas_vec:     [N_MEAS] float GT measurements (MEASUREMENT_KEYS order)
        birads_idx:   int GT BI-RADS class index (-100 / <0 = missing)
        pathology_idx:int GT pathology class index (-100 / <0 = missing)
        """
        d = {}
        if isinstance(full_morphology, dict):
            for _k in REPORT_MORPHOLOGY_KEYS:
                if _k in full_morphology:
                    try:
                        d[_k] = float(full_morphology[_k])
                    except (TypeError, ValueError):
                        pass
        try:
            mv = [float(v) for v in (meas_vec.tolist() if hasattr(meas_vec, "tolist") else meas_vec)]
        except Exception:
            mv = []
        if len(mv) >= 4:
            d["max_diameter_px"] = mv[0]
            d["perpendicular_diameter_px"] = mv[1]
            _bw, _bh = mv[2], mv[3]
            d["orientation"] = ("wider-than-tall orientation" if _bw > _bh
                                else "taller-than-wide orientation" if _bh > _bw
                                else "approximately equal width and height")
        _pi = int(pathology_idx)
        if _pi >= 0:
            _pn = getattr(self, "pathology_class_names", ["benign", "malignant"])
            d["pathology"] = _pn[_pi] if _pi < len(_pn) else str(_pi)
        _bi = int(birads_idx)
        if _bi >= 0:
            _bn = getattr(self, "birads_class_names", None) or ["2", "3", "4", "5"]
            d["birads"] = _bn[_bi] if _bi < len(_bn) else str(_bi)
        return d

    def _build_report_tokens(self, mod_name, active_tags, target_report, use_tag_prompt=True, descriptors=None):
        if descriptors is not None:
            content_str = self._build_descriptor_content(descriptors)
        elif not use_tag_prompt:
            content_str = "Describe the radiological findings. Report:"
        elif active_tags:
            tag_str = ", ".join(active_tags)
            content_str = f"Detected abnormalities: {tag_str}. Report:"
        else:
            content_str = "No abnormality detected. Report:"

        messages = [
            {"role": "system", "content": f"You are an expert {mod_name} radiologist. Describe the findings."},
            {"role": "user", "content": content_str}
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        full_text = prompt_text + " " + target_report + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
        report_ids = self.tokenizer(target_report + self.tokenizer.eos_token, add_special_tokens=False)['input_ids']

        available = self.max_seq_len - len(prompt_ids)

        if available < 10:
            input_ids = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )['input_ids'].squeeze(0)
        else:
            final_ids = prompt_ids + report_ids[:available]
            if len(final_ids) < self.max_seq_len:
                final_ids = final_ids + [self.tokenizer.pad_token_id] * (self.max_seq_len - len(final_ids))
            else:
                final_ids = final_ids[:self.max_seq_len]
            input_ids = torch.tensor(final_ids, dtype=torch.long)

        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = -100

        eos_indices = (labels == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
        report_eos_indices = eos_indices[eos_indices >= prompt_len]

        if len(report_eos_indices) > 0:
            stop_idx = report_eos_indices[0].item()
            if stop_idx + 1 < len(labels):
                labels[stop_idx + 1:] = -100
        else:
            labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels

    def _load_segmentation(self, item):
        mask_tensor = np.zeros(
            (self.num_seg_classes, self.target_size[1], self.target_size[0]),
            dtype=np.float32
        )
        active_channels = np.zeros(self.num_seg_classes, dtype=np.float32)   # NEW

        mask_path = item.get('seg_mask_path')
        local_classes = item.get('seg_local_classes', {})

        if not mask_path or not os.path.exists(mask_path):
            bbox = torch.tensor([0, 0, self.target_size[0], self.target_size[1]], dtype=torch.float32)
            return torch.from_numpy(mask_tensor).float(), bbox, torch.from_numpy(active_channels).float()   # CHANGED

        try:
            mask_pil = Image.open(mask_path)
            mask_pil = mask_pil.resize(self.target_size, resample=Image.NEAREST)
            mask_np = np.array(mask_pil)
            if len(mask_np.shape) > 2:
                mask_np = mask_np[:, :, 0]
        except Exception:
            bbox = torch.tensor([0, 0, self.target_size[0], self.target_size[1]], dtype=torch.float32)
            return torch.from_numpy(mask_tensor).float(), bbox, torch.from_numpy(active_channels).float()   # CHANGED

        ds_name = item['ds_name']

        if ds_name == "LSS_MRI_AISSLab":
            mask_cleaned = mask_np
            is_binary_blob = False
        elif ds_name == "CXLSeg":
            mask_cleaned = np.where(mask_np > 128, 255, 0).astype(np.uint8)
            is_binary_blob = True
        elif ds_name == "BUSI":
            mask_cleaned = np.where(mask_np > 0, 255, 0).astype(np.uint8)
            is_binary_blob = True
        elif ds_name == "BUS_BRA":
            mask_cleaned = np.where(mask_np > 0, 255, 0).astype(np.uint8)
            is_binary_blob = True
        else:
            unique_vals = np.unique(mask_np)
            is_binary_blob = (unique_vals.max() > 100) and (255 in unique_vals)
            if is_binary_blob:
                mask_cleaned = np.where(mask_np > 128, 255, 0).astype(np.uint8)
            else:
                mask_cleaned = mask_np

        for pixel_val_str, class_name in local_classes.items():
            if "background" in class_name.lower():
                continue
            if class_name not in self.seg_class_map:
                continue

            global_ch_idx = self.seg_class_map[class_name]
            active_channels[global_ch_idx] = 1.0                              # NEW

            if is_binary_blob:
                binary_region = (mask_cleaned == 255).astype(np.float32)
            else:
                target_pixel_val = int(pixel_val_str)
                lower = target_pixel_val - 5
                upper = target_pixel_val + 5
                binary_region = ((mask_cleaned >= lower) & (mask_cleaned <= upper)).astype(np.float32)

            mask_tensor[global_ch_idx] = np.maximum(mask_tensor[global_ch_idx], binary_region)

        active_mask = mask_tensor.sum(axis=0) > 0
        if active_mask.any():
            y_ind, x_ind = np.where(active_mask)
            bbox = torch.tensor(
                [np.min(x_ind), np.min(y_ind), np.max(x_ind), np.max(y_ind)],
                dtype=torch.float32
            )
        else:
            bbox = torch.tensor([0, 0, self.target_size[0], self.target_size[1]], dtype=torch.float32)

        return torch.from_numpy(mask_tensor).float(), bbox, torch.from_numpy(active_channels).float()

    def _build_detection_targets(self, item):
        boxes = item.get('det_boxes_raw', [])
        labels = item.get('det_labels_raw', [])
        severity = item.get('det_severity_raw', None)
        label_map = item.get('det_label_map', {})

        if 'orig_size' in item:
            orig_w, orig_h = item['orig_size']
        else:
            try:
                with Image.open(item['img_path']) as im:
                    orig_w, orig_h = im.size
            except Exception:
                orig_w, orig_h = self.target_size

        target_w, target_h = self.target_size
        sx = target_w / float(orig_w)
        sy = target_h / float(orig_h)

        mapped_boxes = []
        mapped_labels = []
        mapped_severity = []

        for i, box in enumerate(boxes):
            if len(box) != 4:
                continue

            xmin, ymin, xmax, ymax = box
            xmin *= sx
            xmax *= sx
            ymin *= sy
            ymax *= sy

            raw_lbl = labels[i]
            lbl_name = label_map.get(str(raw_lbl), None)

            if lbl_name is None or lbl_name not in self.det_class_map:
                continue

            mapped_boxes.append([xmin, ymin, xmax, ymax])
            mapped_labels.append(self.det_class_map[lbl_name])

            if severity is not None and i < len(severity):
                mapped_severity.append(severity[i])

        if len(mapped_boxes) == 0:
            det_targets = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.long),
            }
            if severity is not None:
                det_targets['severity'] = torch.zeros((0,), dtype=torch.long)
            return det_targets

        boxes_xyxy = torch.tensor(mapped_boxes, dtype=torch.float32)
        labels_t = torch.tensor(mapped_labels, dtype=torch.long)

        det_targets = {
            'boxes': boxes_xyxy,  # keep absolute xyxy for ROI / visualization
            'boxes_norm': normalize_boxes(boxes_xyxy, target_w, target_h),  # normalized cxcywh
            'labels': labels_t,
        }
        if severity is not None:
            det_targets['severity'] = torch.tensor(mapped_severity, dtype=torch.long)

        return det_targets

    def _get_cached_segmentation(self, idx, item):
        if not self.cache_targets_in_ram:
            return self._load_segmentation(item)

        out = self.seg_cache.get(idx, None)
        if out is None:
            out = self._load_segmentation(item)
            self.seg_cache[idx] = out
        return out

    def _get_cached_det_targets(self, idx, item):
        if not self.cache_targets_in_ram:
            return self._build_detection_targets(item)

        out = self.det_cache.get(idx, None)
        if out is None:
            out = self._build_detection_targets(item)
            self.det_cache[idx] = out
        return out

    # =========================================================
    # Main fetch
    # =========================================================
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["img_path"]
        tasks = item["tasks"]
        mod_name = item["mod_name"]
        ds_name = item["ds_name"]

        # ------------------------------------------------------------
        # Determine whether joint BUS-BRA augmentation is needed.
        # In training, every BUS-BRA classification sample with an available
        # lesion mask uses the same image/mask augmentation path, even when
        # segmentation is only a dependency and has_seg=False.
        # ------------------------------------------------------------
        def _item_bool(value):
            return (
                bool(value.item())
                if torch.is_tensor(value)
                else bool(value)
            )

        _has_cls_task = _item_bool(item.get("has_cls", False))
        _has_busbra_specialist_label = (
            _has_cls_task
            or _item_bool(item.get("has_birads", False))
            or _item_bool(item.get("has_pathology", False))
        )
        _mask_path = item.get("seg_mask_path")
        _has_usable_busbra_mask = bool(
            _mask_path and os.path.exists(_mask_path)
        )

        is_busbra_joint = (
            ds_name == "BUS_BRA"
            and (
                bool(item["has_seg"].item())
                or (
                    self.split == "train"
                    and _has_busbra_specialist_label
                    and _has_usable_busbra_mask
                )
            )
        )
        is_xray_seg_joint = (
            not is_busbra_joint
            and mod_name == "XRay"
            and bool(item["has_seg"].item())
        )

        image_tensor = None
        seg_mask = None
        bbox = None
        seg_active_ch = None
        augmented_measurements = None
        augmented_measurements_present = None

        # ============================================================
        # CASE 1: BUS-BRA joint image/mask path
        # Load the image and mask before applying transformations.
        # ============================================================
        if is_busbra_joint:

            # Do not retrieve a transformed BUS-BRA training image
            # from the image cache because augmentation must be random
            # on every epoch.
            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
            except Exception as e:
                print(f"[Warning] Failed to load image {img_path}: {e}")
                image = Image.new("RGB", self.target_size)

            # The mask loader already:
            # 1. reads the mask,
            # 2. resizes it with nearest-neighbor interpolation,
            # 3. converts BUS_BRA masks to binary masks,
            # 4. maps them into the global segmentation channels.
            seg_mask, bbox, seg_active_ch = self._get_cached_segmentation(
                idx,
                item,
            )

            # Important when the target comes from the RAM cache.
            # Never transform the cached tensor directly.
            seg_mask = seg_mask.clone()
            seg_active_ch = seg_active_ch.clone()

            # Apply identical geometric transformations to image and mask.
            # Only mild gamma/contrast image perturbations are handled inside this class.
            image_tensor, seg_mask = self.busbra_joint_transform(
                image,
                seg_mask,
            )

            # The old bbox no longer matches after a horizontal flip.
            bbox = self._bbox_from_mask_tensor(seg_mask)

            # Recompute full-field BUS-BRA morphology descriptors after the
            # shared image/mask transform. The lesion crop is never used for
            # descriptor computation.
            if self.split == "train":
                _meas_np, _meas_present = compute_mask_measurements(seg_mask)
                augmented_measurements = torch.from_numpy(
                    _meas_np
                ).to(dtype=torch.float32)
                augmented_measurements_present = torch.tensor(
                    bool(_meas_present),
                    dtype=torch.bool,
                )

        # ============================================================
        # CASE 2: X-ray segmentation with synchronized geometry.
        # ============================================================
        elif is_xray_seg_joint:
            try:
                with Image.open(img_path) as im:
                    image = im.convert("RGB")
            except Exception as e:
                print(f"[Warning] Failed to load image {img_path}: {e}")
                image = Image.new("RGB", self.target_size)

            seg_mask, bbox, seg_active_ch = self._get_cached_segmentation(
                idx,
                item,
            )
            seg_mask = seg_mask.clone()
            seg_active_ch = seg_active_ch.clone()
            image_tensor, seg_mask = self.xray_joint_seg_transform(
                image,
                seg_mask,
            )
            bbox = self._bbox_from_mask_tensor(seg_mask)

        # ============================================================
        # CASE 3: All other datasets.
        # ============================================================
        else:
            if self.cache_images_in_ram:
                image_tensor = self.image_cache.get(img_path, None)

            if image_tensor is None:
                try:
                    with Image.open(img_path) as im:
                        image = im.convert("RGB")
                except Exception as e:
                    print(f"[Warning] Failed to load image {img_path}: {e}")
                    image = Image.new("RGB", self.target_size)

                transform = self._get_transform(
                    mod_name,
                    ds_name,
                )
                image_tensor = transform(image)

                if self.cache_images_in_ram:
                    self.image_cache[img_path] = image_tensor

        # ------------------------------------------------------------
        # Base sample
        # ------------------------------------------------------------
        sample = {
            "image": image_tensor,
            "tasks": tasks,
            "ds_name": ds_name,
            "image_path": img_path,
            "target_class": item.get("target_class", "None"),
            "modality_id": item["modality_id"],
            "report_text": item.get("report_text", ""),
            # (width, height) of the source BUS-BRA mask used for GT morphology.
            "measurement_size": item.get(
                "measurement_size",
                item.get("orig_size", self.target_size),
            ),

            "has_seg": item["has_seg"],
            "has_cls": item["has_cls"],
            "has_rep": item["has_rep"],
            "has_det": item["has_det"],
            "task_mask": item["task_mask"],
            "is_lss_mri": item["is_lss_mri"],
        }

        # ------------------------------------------------------------
        # Classification
        # ------------------------------------------------------------
        _raw_has_cls = item.get("has_cls", False)

        if torch.is_tensor(_raw_has_cls):
            has_cls_task = bool(_raw_has_cls.item())
        else:
            has_cls_task = bool(_raw_has_cls)

        has_generic_cls = (
            has_cls_task
            and "class_labels" in item
            and "class_mask" in item
        )

        # This flag must represent generic classification only.
        sample["has_cls"] = torch.tensor(
            has_generic_cls,
            dtype=torch.bool,
        )

        if has_generic_cls:
            sample["class_labels"] = item["class_labels"]
            sample["class_mask"] = item["class_mask"]

            sample["cls_problem_type"] = item.get(
                "cls_problem_type",
                torch.tensor(0, dtype=torch.long),
            )

            sample["class_index"] = item.get(
                "class_index",
                torch.tensor(-100, dtype=torch.long),
            )

            sample["local_class_indices"] = item.get(
                "local_class_indices",
                torch.nonzero(
                    item["class_mask"] > 0,
                    as_tuple=False,
                ).view(-1).long(),
            )

        # ------------------------------------------------------------
        # Report generation
        # ------------------------------------------------------------
        if item["has_rep"].item() and self.tokenizer is not None:
            sample["report_ids"] = item["report_ids"]
            sample["report_labels"] = item["report_labels"]

        # ------------------------------------------------------------
        # BUS-BRA descriptor GT fields — always included when present.
        # These are needed at val/test for specialist report/cls mode
        # (use_gt_descriptors=True) and for collation (always stacked).
        # They are built unconditionally from the annotation at dataset
        # init (not gated on tasks) so they are always in `item`.
        # ------------------------------------------------------------
        sample["has_birads"]          = item.get("has_birads",
                                            torch.tensor(False, dtype=torch.bool))
        sample["birads_index"]        = item.get("birads_index",
                                            torch.tensor(-100, dtype=torch.long))
        sample["birads_ordinal"]      = item.get("birads_ordinal",
                                            torch.zeros(max(self.birads_num_classes - 1, 1),
                                                        dtype=torch.float32))
        if augmented_measurements is not None:
            sample["measurements"] = augmented_measurements
            sample["measurements_present"] = augmented_measurements_present
        else:
            sample["measurements"] = item.get(
                "measurements",
                torch.zeros(
                    len(MEASUREMENT_KEYS),
                    dtype=torch.float32,
                ),
            )
            sample["measurements_present"] = item.get(
                "measurements_present",
                torch.tensor(False, dtype=torch.bool),
            )
        sample["pathology_index"]     = item.get("pathology_index",
                                            torch.tensor(-100, dtype=torch.long))
        sample["report_morphology_gt"] = item.get("report_morphology_gt", {})
        sample["busbra_seg_channel_indices"] = item.get("busbra_seg_channel_indices", tuple())
        sample["has_pathology"]      = item.get("has_pathology",
                                            torch.tensor(False, dtype=torch.bool))

        # ------------------------------------------------------------
        # Segmentation
        # ------------------------------------------------------------
        if item["has_seg"].item():

            # BUS-BRA mask was already loaded and augmented above.
            if not (is_busbra_joint or is_xray_seg_joint):
                seg_mask, bbox, seg_active_ch = (
                    self._get_cached_segmentation(idx, item)
                )

            sample["seg_mask"] = seg_mask
            sample["bbox"] = bbox
            sample["seg_active_ch"] = seg_active_ch

        # ------------------------------------------------------------
        # MRI detection
        # Load det_targets when has_det=True (supervised) OR when
        # detection is a dependency task (task_mask[3]=1) so the
        # DINO forward receives valid (empty-box) targets even in
        # specialist report mode where has_det=False.
        # ------------------------------------------------------------
        _det_is_dep = (float(item["task_mask"][3].item()) > 0.5)
        if (
            (item["has_det"].item() or _det_is_dep)
            and ds_name in self.full_slice_mri_datasets
        ):
            sample["det_targets"] = self._get_cached_det_targets(
                idx,
                item,
            )

        return sample
    
    def _bbox_from_mask_tensor(self, mask):
        """
        mask: [C, H, W]
        """
        foreground = mask.sum(dim=0) > 0

        if foreground.any():
            ys, xs = torch.where(foreground)

            return torch.stack([
                xs.min(),
                ys.min(),
                xs.max(),
                ys.max(),
            ]).float()

        return torch.tensor(
            [0, 0, self.target_size[0], self.target_size[1]],
            dtype=torch.float32,
        )


def unified_collate(batch):
    batch_out = {
        'image': torch.stack([b['image'] for b in batch]),
        'tasks': [b['tasks'] for b in batch],
        'target_class': [b.get('target_class', 'None') for b in batch],
        'modality_id': torch.stack([b['modality_id'] for b in batch]),
        'report_text': [b.get('report_text', "") for b in batch],
        'ds_name': [b.get('ds_name', 'Unknown') for b in batch],
        'image_path': [b.get('image_path', '') for b in batch]
    }

    B = len(batch)
    H, W = batch[0]['image'].shape[-2:]
    batch_out['measurement_sizes'] = [
        b.get('measurement_size', (W, H)) for b in batch
    ]
    batch_out['busbra_seg_channel_indices'] = [
        b.get('busbra_seg_channel_indices', tuple()) for b in batch
    ]
    batch_out['report_morphology_gt'] = [
        b.get('report_morphology_gt', {}) for b in batch
    ]

    # ------------------------------------------------------------
    # Use precomputed flags directly
    # ------------------------------------------------------------
    is_lss_mri = torch.stack([b['is_lss_mri'] for b in batch])
    has_seg = torch.stack([b['has_seg'] for b in batch])
    has_cls = torch.stack([b['has_cls'] for b in batch])
    has_rep = torch.stack([b['has_rep'] for b in batch])
    has_det = torch.stack([b['has_det'] for b in batch])
    task_mask = torch.stack([b['task_mask'] for b in batch])

    batch_out['is_lss_mri'] = is_lss_mri
    batch_out['has_seg'] = has_seg
    batch_out['has_cls'] = has_cls
    batch_out['has_rep'] = has_rep
    batch_out['has_det'] = has_det
    batch_out['task_mask'] = task_mask
    # ---- BUS-BRA: BI-RADS + measurements (added) ----
    _K1 = batch[0]['birads_ordinal'].shape[0] if 'birads_ordinal' in batch[0] else 3
    batch_out['has_birads'] = torch.stack([
        b.get('has_birads', torch.tensor(False)) for b in batch])
    batch_out['birads_index'] = torch.stack([
        b.get('birads_index', torch.tensor(-100, dtype=torch.long)) for b in batch])
    batch_out['birads_ordinal'] = torch.stack([
        b.get('birads_ordinal', torch.zeros(_K1, dtype=torch.float32)) for b in batch])
    batch_out['measurements'] = torch.stack([
        b.get('measurements', torch.zeros(len(MEASUREMENT_KEYS), dtype=torch.float32)) for b in batch])
    batch_out['measurements_present'] = torch.stack([
        b.get('measurements_present', torch.tensor(False)) for b in batch])
    batch_out['pathology_index'] = torch.stack([
        b.get('pathology_index', torch.tensor(-100, dtype=torch.long)) for b in batch])
    batch_out['has_pathology'] = torch.stack([
        b.get('has_pathology', torch.tensor(False, dtype=torch.bool)) for b in batch])

    # ----------------
    # segmentation
    # ----------------
    if has_seg.any():
        seg_channels = next(
            (b['seg_mask'].shape[0] for b in batch if 'seg_mask' in b),
            1
        )
        batch_out['seg_mask'] = torch.stack([
            b.get('seg_mask', torch.zeros((seg_channels, H, W), dtype=torch.float32))
            for b in batch
        ])
        batch_out['bbox'] = torch.stack([
            b.get('bbox', torch.tensor([0, 0, 1, 1], dtype=torch.float32))
            for b in batch
        ])
        batch_out['seg_active_ch'] = torch.stack([
            b.get('seg_active_ch', torch.zeros(seg_channels, dtype=torch.float32))
            for b in batch
        ])

    # ----------------
    # classification
    # ----------------
    if has_cls.any():
        lbl_size = next(
            (b['class_labels'].shape[0] for b in batch if 'class_labels' in b),
            0
        )
        batch_out['class_labels'] = torch.stack([
            b.get('class_labels', torch.zeros(lbl_size, dtype=torch.float32))
            for b in batch
        ])
        batch_out['class_mask'] = torch.stack([
            b.get('class_mask', torch.zeros(lbl_size, dtype=torch.float32))
            for b in batch
        ])
        max_local = max(
            (b['local_class_indices'].shape[0] for b in batch if 'local_class_indices' in b),
            default=0
        )

        batch_out['cls_problem_type'] = torch.stack([
            b.get('cls_problem_type', torch.tensor(0, dtype=torch.long))
            for b in batch
        ])

        batch_out['class_index'] = torch.stack([
            b.get('class_index', torch.tensor(-100, dtype=torch.long))
            for b in batch
        ])

        batch_out['local_class_indices'] = torch.stack([
            torch.cat([
                b.get('local_class_indices', torch.empty(0, dtype=torch.long)),
                torch.full(
                    (max_local - b.get('local_class_indices', torch.empty(0, dtype=torch.long)).shape[0],),
                    -1,
                    dtype=torch.long
                )
            ])
            for b in batch
        ])

    # ----------------
    # report
    # ----------------
    if has_rep.any():
        rep_len = next(
            (b['report_ids'].shape[0] for b in batch if 'report_ids' in b),
            60
        )
        batch_out['report_ids'] = torch.stack([
            b.get('report_ids', torch.zeros(rep_len, dtype=torch.long))
            for b in batch
        ])
        batch_out['report_labels'] = torch.stack([
            b.get('report_labels', torch.full((rep_len,), -100, dtype=torch.long))
            for b in batch
        ])

    # ----------------
    # detection / MRI ROI supervision
    # ----------------
    # Build det_targets if has_det is active OR if detection is a
    # dependency task (task_mask[:,3]>0.5) for specialist report mode.
    _det_mask_any = has_det.any()
    _task_masks = torch.stack([b["task_mask"] for b in batch])  # [B,4]
    _det_dep_any = (_task_masks[:, 3] > 0.5).any()
    if _det_mask_any or _det_dep_any:
        det_targets = []
        gt_boxes_for_roi = []
        gt_level_labels_for_roi = []
        gt_severity_for_roi = []
        mri_visible_level_mask = []

        for i, b in enumerate(batch):
            t = b.get('det_targets', None)

            if t is None:
                t = {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros((0,), dtype=torch.long),
                    'severity': torch.zeros((0,), dtype=torch.long)
                }
            else:
                if 'severity' not in t:
                    t = {
                        'boxes': t['boxes'],
                        'labels': t['labels'],
                        'severity': torch.zeros((t['boxes'].shape[0],), dtype=torch.long)
                    }

            det_targets.append(t)

            if has_det[i] and t['boxes'].numel() > 0:
                gt_boxes_for_roi.append(t['boxes'])
                gt_level_labels_for_roi.append(t['labels'])
                gt_severity_for_roi.append(t['severity'])
                mri_visible_level_mask.append(torch.ones_like(t['labels'], dtype=torch.bool))
            else:
                gt_boxes_for_roi.append(torch.zeros((0, 4), dtype=torch.float32))
                gt_level_labels_for_roi.append(torch.zeros((0,), dtype=torch.long))
                gt_severity_for_roi.append(torch.zeros((0,), dtype=torch.long))
                mri_visible_level_mask.append(torch.zeros((0,), dtype=torch.bool))

        batch_out['det_targets'] = det_targets
        batch_out['gt_boxes_for_roi'] = gt_boxes_for_roi
        batch_out['gt_level_labels_for_roi'] = gt_level_labels_for_roi
        batch_out['gt_severity_for_roi'] = gt_severity_for_roi
        batch_out['mri_visible_level_mask'] = mri_visible_level_mask

    return batch_out

def normalize(text):
            # Lowercase and remove punctuation (keep only words and numbers)
            return re.sub(r'[^\w\s]', '', text).lower().strip()

def box_iou_matrix_xyxy(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
                (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
                (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)

def match_detection_predictions_to_gt_class_aware(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    Greedy class-aware matching for metrics.
    Returns list of (pred_idx, gt_idx, iou)
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    ious = box_iou_matrix_xyxy(pred_boxes, gt_boxes)
    matches = []

    used_pred = set()
    used_gt = set()

    # candidate pairs only when class matches
    candidates = []
    for p in range(pred_boxes.shape[0]):
        for g in range(gt_boxes.shape[0]):
            if int(pred_labels[p].item()) != int(gt_labels[g].item()):
                continue
            iou = float(ious[p, g].item())
            if iou >= iou_thresh:
                candidates.append((iou, p, g))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, p, g in candidates:
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(p)
        used_gt.add(g)
        matches.append((p, g, iou))

    return matches

def match_roi_boxes_to_gt_class_aware(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh=0.5):
    """
    Greedy class-aware matching for MRI ROI severity.
    Returns list of (pred_idx, gt_idx, iou)
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    ious = box_iou_matrix_xyxy(pred_boxes, gt_boxes)
    candidates = []

    for p in range(pred_boxes.shape[0]):
        for g in range(gt_boxes.shape[0]):
            if int(pred_labels[p].item()) != int(gt_labels[g].item()):
                continue
            iou = float(ious[p, g].item())
            if iou >= iou_thresh:
                candidates.append((iou, p, g))

    candidates.sort(reverse=True, key=lambda x: x[0])

    used_pred = set()
    used_gt = set()
    matches = []

    for iou, p, g in candidates:
        if p in used_pred or g in used_gt:
            continue
        used_pred.add(p)
        used_gt.add(g)
        matches.append((p, g, iou))

    return matches

def gather_mri_roi_matched_samples(outputs, det_targets, batch_tasks, device, iou_thresh=0.5):
    if (
        outputs.get("mri_roi_logits", None) is None or
        outputs.get("mri_roi_batch_idx", None) is None or
        outputs.get("mri_roi_box_idx", None) is None or
        outputs.get("mri_pred_boxes", None) is None or
        outputs.get("mri_pred_labels", None) is None or
        det_targets is None
    ):
        return None, None

    roi_logits = outputs["mri_roi_logits"].float()
    roi_batch_idx = outputs["mri_roi_batch_idx"]
    roi_box_idx = outputs["mri_roi_box_idx"]
    pred_boxes_list = outputs["mri_pred_boxes"]
    pred_labels_list = outputs["mri_pred_labels"]

    all_logits = []
    all_targets = []

    B = len(batch_tasks)
    for b in range(B):
        if 'detection' not in str(batch_tasks[b]).lower():
            continue

        gt_boxes_b = det_targets[b]["boxes"]
        gt_labels_b = det_targets[b]["labels"]
        gt_sev_b = det_targets[b].get("severity", None)

        if gt_sev_b is None or gt_boxes_b.numel() == 0:
            continue

        pred_mask = (roi_batch_idx == b)
        if pred_mask.sum() == 0:
            continue

        pred_logits_b = roi_logits[pred_mask]
        pred_local_idx_b = roi_box_idx[pred_mask]
        pred_boxes_full_b = pred_boxes_list[b]
        pred_labels_full_b = pred_labels_list[b]

        if pred_boxes_full_b is None or pred_boxes_full_b.numel() == 0:
            continue
        if pred_logits_b.ndim != 2:
            continue

        valid_local = (
            (pred_local_idx_b >= 0) &
            (pred_local_idx_b < pred_boxes_full_b.shape[0])
        )
        if not valid_local.any():
            continue

        pred_logits_b = pred_logits_b[valid_local]
        pred_local_idx_b = pred_local_idx_b[valid_local]
        pred_boxes_b = pred_boxes_full_b[pred_local_idx_b]
        pred_level_labels_b = pred_labels_full_b[pred_local_idx_b]

        gt_sev_b = gt_sev_b.long()
        valid_gt = (gt_sev_b >= 0) & (gt_sev_b < pred_logits_b.shape[1])
        if not valid_gt.any():
            continue

        gt_boxes_b = gt_boxes_b[valid_gt]
        gt_labels_b = gt_labels_b[valid_gt]
        gt_sev_b = gt_sev_b[valid_gt]

        if gt_boxes_b.numel() == 0:
            continue

        matches = match_roi_boxes_to_gt_class_aware(
            pred_boxes_b,
            pred_level_labels_b,
            gt_boxes_b,
            gt_labels_b,
            iou_thresh=iou_thresh,
        )
        if len(matches) == 0:
            continue

        pred_ids = torch.tensor([m[0] for m in matches], device=device, dtype=torch.long)
        gt_ids = torch.tensor([m[1] for m in matches], device=device, dtype=torch.long)

        all_logits.append(pred_logits_b[pred_ids])
        all_targets.append(gt_sev_b[gt_ids])

    if len(all_logits) == 0:
        return None, None

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)

def get_det_predictions_for_eval(model, outputs, imgs, score_thresh=0.1, topk=8, nms_iou=0.4):
    """
    Returns per-image detection predictions in the same format:
      pred_boxes_list, pred_labels_list, pred_scores_list, pred_query_ids_list
    """
    if outputs.get("detection", None) is None:
        B = imgs.shape[0]
        empty_boxes = [torch.zeros((0, 4), device=imgs.device) for _ in range(B)]
        empty_labels = [torch.zeros((0,), dtype=torch.long, device=imgs.device) for _ in range(B)]
        empty_scores = [torch.zeros((0,), device=imgs.device) for _ in range(B)]
        empty_qids = [torch.zeros((0,), dtype=torch.long, device=imgs.device) for _ in range(B)]
        return empty_boxes, empty_labels, empty_scores, empty_qids

    # For now keep the same raw-query decoder for both paths,
    # because your ROI fusion needs query_ids.
    return model.decode_mri_det_predictions(
        outputs["detection"],
        image_size=imgs.shape[-2:],
        score_thresh=score_thresh,
        topk=topk,
        nms_iou=nms_iou
    )

def plot_multiclass_roc_curve(
    y_true,
    y_prob,
    class_names,
    save_path,
    title="Multiclass ROC Curve",
):
    """
    y_true: [N] integer labels
    y_prob: [N, C] softmax probabilities
    class_names: list[str] length C
    """

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    num_classes = y_prob.shape[1]

    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(8, 7))

    class_auc_list = []

    for c in range(num_classes):
        # ROC needs both positive and negative samples for this class
        if len(np.unique(y_true_bin[:, c])) < 2:
            print(f"[ROC] Skipping class {c} ({class_names[c]}) because only one label type exists.")
            class_auc_list.append(np.nan)
            continue

        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        class_auc_list.append(float(roc_auc))

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_names[c]} AUC = {roc_auc:.4f}"
        )

    try:
        macro_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
    except Exception as e:
        print(f"[ROC] Could not compute macro AUC: {e}")
        macro_auc = np.nan

    # Calculate micro AUC, but do NOT plot it
    try:
        fpr_micro, tpr_micro, _ = roc_curve(
            y_true_bin.ravel(),
            y_prob.ravel()
        )
        micro_auc = auc(fpr_micro, tpr_micro)
    except Exception as e:
        print(f"[ROC] Could not compute micro AUC: {e}")
        micro_auc = np.nan
    
    try:
        weighted_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="weighted"
        )
    except Exception as e:
        print(f"[ROC] Could not compute weighted AUC: {e}")
        weighted_auc = np.nan

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=2,
        label="Random baseline"
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        f"{title}\nMacro AUC = {macro_auc:.4f} | Weighted AUC = {weighted_auc:.4f}"
    )

    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "class_auc": class_auc_list,
        "macro_auc": float(macro_auc) if not np.isnan(macro_auc) else 0.0,
        "weighted_auc": float(weighted_auc) if not np.isnan(weighted_auc) else 0.0,
        "micro_auc": float(micro_auc) if not np.isnan(micro_auc) else 0.0,
        "save_path": save_path,
    }

def make_report_prompt_content(active_tags=None, use_guidance=True):
    if not use_guidance:
        return "Describe the radiological findings. Report:"

    active_tags = active_tags or []
    if active_tags:
        return f"Detected abnormalities: {', '.join(active_tags)}. Report:"
    return "No abnormality detected. Report:"

def build_busbra_predicted_prompt(
    outputs,
    sample_idx,
    dataset,
):
    """
    Build a BUS-BRA report prompt entirely from predictions.

    Sources:
        measurements  -> predicted segmentation mask
        orientation   -> predicted mask measurements
        pathology     -> pathology_meas_head
        BI-RADS       -> birads_head
    """

    desc = {}

    # ------------------------------------------------------
    # Predicted measurements from predicted segmentation
    # ------------------------------------------------------
    measurements = outputs.get(
        "report_measurements_used",
        outputs.get("measurements_used", None),
    )
    measurements_present = outputs.get(
        "report_measurements_present_used",
        outputs.get("measurements_present_used", None),
    )

    measurement_is_valid = measurements is not None

    if measurements_present is not None:
        measurement_is_valid = (
            measurement_is_valid
            and bool(measurements_present[sample_idx].item())
        )

    if measurement_is_valid:
        # Preferred path: full source-space morphology computed from the predicted
        # mask after resizing to the original BUS-BRA mask dimensions.
        full_morphology = outputs.get("report_morphology_used", None)
        if isinstance(full_morphology, (list, tuple)) and sample_idx < len(full_morphology):
            _fm = full_morphology[sample_idx]
            if isinstance(_fm, dict):
                for _k in REPORT_MORPHOLOGY_KEYS:
                    if _k in _fm:
                        desc[_k] = float(_fm[_k])

        # Backward-compatible vector fallback for older model files.
        values = (
            measurements[sample_idx]
            .detach()
            .float()
            .cpu()
            .tolist()
        )
        for _i, _k in enumerate(MEASUREMENT_KEYS):
            if _k not in desc and _i < len(values):
                desc[_k] = float(values[_i])

        bbox_width = float(desc.get("bbox_width_px", 0.0))
        bbox_height = float(desc.get("bbox_height_px", 0.0))
        if bbox_width > bbox_height:
            desc["orientation"] = "wider-than-tall orientation"
        elif bbox_height > bbox_width:
            desc["orientation"] = "taller-than-wide orientation"
        elif bbox_width > 0 or bbox_height > 0:
            desc["orientation"] = "approximately equal width and height"

    # ------------------------------------------------------
    # Predicted pathology
    # ------------------------------------------------------
    pathology_logits = outputs.get(
        "pathology_meas_logits",
        None,
    )

    if pathology_logits is not None:
        pathology_names = getattr(
            dataset,
            "pathology_class_names",
            ["benign", "malignant"],
        )

        pathology_idx = int(
            pathology_logits[sample_idx]
            .float()
            .argmax()
            .item()
        )

        if pathology_idx < len(pathology_names):
            desc["pathology"] = pathology_names[pathology_idx]
        else:
            desc["pathology"] = str(pathology_idx)

    # ------------------------------------------------------
    # Predicted BI-RADS
    # ------------------------------------------------------
    birads_logits = outputs.get(
        "birads_logits",
        None,
    )

    if birads_logits is not None:
        birads_names = getattr(
            dataset,
            "birads_class_names",
            None,
        )

        if not birads_names:
            birads_names = ["2", "3", "4", "5"]

        birads_idx = int(
            birads_decode(
                birads_logits[
                    sample_idx:sample_idx + 1
                ].float()
            )[0].item()
        )

        if birads_idx < len(birads_names):
            desc["birads"] = birads_names[birads_idx]
        else:
            desc["birads"] = str(birads_idx)

    content = dataset._build_descriptor_content(
        desc, enforce_pixel_units=True
    )

    return content, desc

def validate(model, loader, device, loss_fns, manager=None, epoch=0, tokenizer=None, max_val_samples=None, run_heavy_metrics=False, roi_cls_weights=None, max_batches=None, pos_weight=None):
    model.eval()
    losses = defaultdict(list)
    # Accumulators
    all_preds_text, all_refs_text = [], []
    all_report_ds_names = []
    all_masks_cls, all_ds_names , all_targets_cls, all_preds_probs = [], [], [], []
    # Detection accumulators (MRI only)
    all_det_pred_labels = []
    all_det_gt_labels = []
    all_det_ious = []
    all_det_scores = []
    all_det_tp = 0
    all_det_fp = 0
    all_det_fn = 0
    seg_dice_by_ds = defaultdict(list)          # per-dataset Dice (added)
    seg_dice_lesion_by_ds = defaultdict(list)   # lesion-present only (added)
    seg_base_dice_by_ds = defaultdict(list)
    birads_pred_all, birads_gt_all, birads_prob_all = [], [], []  # BI-RADS ordinal
    birads_base_pred_all, birads_base_prob_all = [], []
    birads_q_all, birads_base_q_all = [], []
    pathology_pred_all, pathology_gt_all, pathology_prob_all = [], [], []  # BUS-BRA pathology (added)
    pathology_base_pred_all, pathology_base_prob_all = [], []
    multiclass_val_data = defaultdict(lambda: {
        "targets": [],
        "preds": [],
        "probs": [],
    })
    # MRI ROI severity accumulators
    # all_mri_roi_targets = []
    # all_mri_roi_preds = []
    
    class_names = loader.dataset.class_names
    id2mod = {i: name for name, i in loader.dataset.modality_map.items()}
    # Setup
    first_batch_processed = False
    rouge_evaluator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    det_num_classes = len(loader.dataset.det_class_names) if hasattr(loader.dataset, "det_class_names") else 5
    use_prompt_guidance = bool(getattr(model, "use_cross_roi_reasoning", True))

    # mri_roi_class_weights = estimate_mri_severity_class_weights(loader, num_classes=4).to(device, non_blocking=True)
    # print("MRI ROI class weights:", mri_roi_class_weights.tolist())
    
    # Only build the local criterion for the custom-DINO path.
    # Official-DINO path should use model.detector.compute_loss(...)
    det_criterion = None
    if not getattr(model, "use_official_dino", False):
        num_decoder_layers = getattr(model, "det_num_decoder_layers", 6)
        det_criterion = build_detection_criterion(
            det_num_classes=det_num_classes,
            device=device,
            num_decoder_layers=num_decoder_layers,
        )
        
    # Canonical Label
    NORMAL_LABEL = "No_Finding"
    val_rep=False # make True #change
    # val_use_pred_boxes_for_mri_cls = False
    W_SEG, W_CLS, W_REP = 2, 1.0, 1.0
    W_DET = 1.0
    with torch.no_grad():
        for _eval_bi, batch in enumerate(loader):
            if max_batches is not None and _eval_bi >= max_batches:
                break
            loss_terms = []
            weight_terms = []
            dev = move_multitask_batch_to_device(batch, device)
            imgs = dev["imgs"]
            mod_ids = dev["mod_ids"]
            captions = dev["captions"]
            report_labels = dev["report_labels"]
            task_mask = dev["task_mask"]
            has_seg = dev["has_seg"]
            has_cls = dev["has_cls"]
            has_rep = dev["has_rep"]
            has_det = dev["has_det"]
            det_targets = dev["det_targets"]
            gt_boxes_for_roi = dev["gt_boxes_for_roi"]
            is_lss_mri = dev["is_lss_mri"]
    
            B = imgs.shape[0] 
            use_amp = (device == "cuda")
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # 1. Forward Pass (Get Logits & Losses)
                outputs = model(imgs, modality_ids=mod_ids, boxes=gt_boxes_for_roi, det_targets=det_targets, task_mask=task_mask, captions=captions, 
                                report_labels=report_labels, use_pred_boxes_for_mri_cls=False, det_score_thresh=0.1, is_lss_mri=is_lss_mri,
                                # GT descriptors are consumed only when model.use_gt_descriptors=True (specialist mode);
                                # in unified eval the forward ignores these and uses predicted masks.
                                measurements=(dev["measurements"] if "measurements" in dev else (batch["measurements"].to(device, non_blocking=True) if "measurements" in batch else None)),
                                measurements_present=(dev["measurements_present"] if "measurements_present" in dev else (batch["measurements_present"].to(device, non_blocking=True) if "measurements_present" in batch else None)),
                                measurement_sizes=dev.get("measurement_sizes"),
                                busbra_seg_channel_indices=dev.get("busbra_seg_channel_indices"),
                                has_birads=dev.get("has_birads"),
                                has_pathology=dev.get("has_pathology"),
                                seg_mask_gt=(batch["seg_mask"].to(device, non_blocking=True) if "seg_mask" in batch else None),
                                gt_birads_idx=(batch["birads_index"].to(device, non_blocking=True) if "birads_index" in batch else None),
                                gt_pathology_idx=(batch["pathology_index"].to(device, non_blocking=True) if "pathology_index" in batch else None),
                                compute_gradif_reference=True)
                batch_loss_sum = torch.tensor(0.0, device=device)
            
            # --- Segmentation Loss --- #change
            pred_masks_small = outputs.get("pred_masks_small", None)
            seg_pred_batch_idx = outputs.get("seg_pred_batch_idx", None)

            if pred_masks_small is not None and seg_pred_batch_idx is not None and 'seg_mask' in batch and has_seg.any():
                # Among the segmentation-produced subset, keep only samples that truly have GT masks
                seg_supervised_local = has_seg[seg_pred_batch_idx]   # [N_seg_subset]

                if seg_supervised_local.any():
                    seg_gt_batch_idx = seg_pred_batch_idx[seg_supervised_local]  # original batch indices

                    preds = nn.functional.interpolate(
                        pred_masks_small[seg_supervised_local],
                        size=imgs.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                    targets = batch['seg_mask'].to(device, non_blocking=True).float()[seg_gt_batch_idx]
                    active_ch = batch['seg_active_ch'].to(device, non_blocking=True).float()[seg_gt_batch_idx]
                    l_seg = masked_bce_with_logits(preds, targets, active_ch) + loss_fns['seg_dice'](preds, targets, active_channels=active_ch)

                    batch_loss_sum += W_SEG * l_seg
                    losses['seg_loss'].append(l_seg.item())
                    loss_terms.append(W_SEG * l_seg)
                    weight_terms.append(W_SEG)

                    # ---- per-dataset Dice (added) ----
                    with torch.no_grad():
                        _p = (torch.sigmoid(preds) > 0.5).float()
                        _t = (targets > 0.5).float()
                        _inter = (_p * _t).sum(dim=(2, 3))
                        _den = _p.sum(dim=(2, 3)) + _t.sum(dim=(2, 3))
                        _dice_ch = (2.0 * _inter + 1e-6) / (_den + 1e-6)                  # [N, C]
                        _a = active_ch
                        _dice_samp = (_dice_ch * _a).sum(dim=1) / (_a.sum(dim=1) + 1e-8)  # [N]
                        _has_lesion = (_t.sum(dim=(1, 2, 3)) > 0)                          # [N]
                        _base_dice_samp = None
                        _base_small = outputs.get("pred_masks_base_small", None)
                        if _base_small is not None:
                            _base_preds = nn.functional.interpolate(
                                _base_small[seg_supervised_local],
                                size=imgs.shape[-2:],
                                mode='bilinear',
                                align_corners=False,
                            )
                            _bp = (torch.sigmoid(_base_preds) > 0.5).float()
                            _binter = (_bp * _t).sum(dim=(2, 3))
                            _bden = _bp.sum(dim=(2, 3)) + _t.sum(dim=(2, 3))
                            _bdice_ch = (2.0 * _binter + 1e-6) / (_bden + 1e-6)
                            _base_dice_samp = (
                                (_bdice_ch * _a).sum(dim=1) / (_a.sum(dim=1) + 1e-8)
                            )
                    _seg_ds_names = [batch['ds_name'][int(_i)] for _i in seg_gt_batch_idx.tolist()]
                    for _local_i, (_nm, _dv, _hl) in enumerate(zip(
                        _seg_ds_names, _dice_samp.tolist(), _has_lesion.tolist()
                    )):
                        seg_dice_by_ds[_nm].append(float(_dv))
                        if _hl:
                            seg_dice_lesion_by_ds[_nm].append(float(_dv))
                        if _base_dice_samp is not None:
                            seg_base_dice_by_ds[_nm].append(
                                float(_base_dice_samp[_local_i].item())
                            )

            # --- Classification Loss & Accumulation --- #change
            if outputs.get('class_logits') is not None and 'class_labels' in batch:
                logits_all = outputs['class_logits'].float()
                labels_all = batch['class_labels'].to(device, non_blocking=True).float()
                mask_all = batch['class_mask'].to(device, non_blocking=True).float()
                cls_problem_type = batch['cls_problem_type'].to(device, non_blocking=True)
                class_index_all = batch['class_index'].to(device, non_blocking=True)
                local_class_indices_all = batch['local_class_indices'].to(device, non_blocking=True)

                cls_terms = []

                # multilabel
                ml_mask = has_cls & (cls_problem_type == 0)
                if ml_mask.any():
                    logits_ml = logits_all[ml_mask]
                    targets_ml = labels_all[ml_mask]
                    mask_ml = mask_all[ml_mask]

                    mask_bool = mask_ml.bool()
                    masked_logits = logits_ml.clone()
                    masked_logits[~mask_bool] = -1e9

                    per_label_loss = loss_fns['cls_focal'](logits_ml, targets_ml)
                    masked_loss = per_label_loss * mask_ml
                    l_cls_ml = masked_loss.sum() / (mask_ml.sum() + 1e-8)

                    batch_loss_sum += W_CLS * l_cls_ml
                    losses['cls_multilabel_loss'].append(l_cls_ml.item())
                    cls_terms.append(l_cls_ml)

                    probs_ml = torch.sigmoid(masked_logits)
                    all_preds_probs.append(probs_ml.cpu().numpy())
                    all_targets_cls.append(targets_ml.cpu().numpy())
                    all_masks_cls.append(mask_ml.cpu().numpy())
                    all_ds_names.extend([batch['ds_name'][i] for i in torch.where(ml_mask)[0]])

                # multiclass
                mc_mask = has_cls & (cls_problem_type == 1)
                if mc_mask.any():
                    mc_losses = []

                    for bi in torch.where(mc_mask)[0].tolist():
                        idxs = local_class_indices_all[bi]
                        idxs = idxs[idxs >= 0].long()
                        if idxs.numel() == 0:
                            continue

                        logits_i = logits_all[bi, idxs].unsqueeze(0)
                        target_i = class_index_all[bi].view(1)
                        if int(target_i.item()) < 0:
                            continue

                        probs_i = torch.softmax(logits_i, dim=1).squeeze(0)
                        pred_i = probs_i.argmax().item()
                        ds_i = batch['ds_name'][bi]

                        if ds_i == "LSS_MRI_ROI_CLS":
                            loss_i = F.cross_entropy(logits_i, target_i, weight=roi_cls_weights)
                        else:
                            loss_i = F.cross_entropy(logits_i, target_i)

                        mc_losses.append(loss_i)

                        multiclass_val_data[ds_i]["targets"].append(int(target_i.item()))
                        multiclass_val_data[ds_i]["preds"].append(int(pred_i))
                        multiclass_val_data[ds_i]["probs"].append(probs_i.detach().cpu().numpy())

                    if len(mc_losses) > 0:
                        l_cls_mc = torch.stack(mc_losses).mean()
                        batch_loss_sum += W_CLS * l_cls_mc
                        losses['cls_multiclass_loss'].append(l_cls_mc.item())
                        cls_terms.append(l_cls_mc)

                if len(cls_terms) > 0:
                    losses['cls_loss'].append(torch.stack(cls_terms).sum().item())

            # --- Report Loss --- #change
            if outputs.get('rep_loss') is not None and captions is not None:
                if has_rep.any():
                    l_rep = outputs['rep_loss']
                    batch_loss_sum += W_REP * l_rep
                    losses['rep_loss'].append(l_rep.item())
                    loss_terms.append(W_REP * l_rep)
                    weight_terms.append(W_REP)
            
            # --- BI-RADS (ordinal) accumulation (added) ---
            if outputs.get('birads_logits') is not None and 'has_birads' in batch:
                _hb = batch['has_birads']
                if _hb.any():
                    _blogits = outputs['birads_logits'].float()
                    _thr = _model_birads_thresholds(model, _blogits.device)
                    _bpred = birads_decode(_blogits, _thr).cpu()
                    _bprob = birads_class_probabilities(_blogits).cpu()
                    _bq = torch.sigmoid(_blogits).cpu()
                    _bgt = batch['birads_index']
                    _base_blogits = outputs.get('birads_base_logits', None)
                    if _base_blogits is not None:
                        _base_bpred = birads_decode(_base_blogits.float(), _thr).cpu()
                        _base_bprob = birads_class_probabilities(
                            _base_blogits.float()
                        ).cpu()
                        _base_bq = torch.sigmoid(_base_blogits.float()).cpu()
                    else:
                        _base_bpred = _bpred
                        _base_bprob = _bprob
                        _base_bq = _bq
                    for _bi2 in _hb.nonzero(as_tuple=True)[0].tolist():
                        _g = int(_bgt[_bi2])
                        if _g < 0:
                            continue
                        birads_pred_all.append(int(_bpred[_bi2]))
                        birads_gt_all.append(_g)
                        birads_prob_all.append(_bprob[_bi2].numpy())
                        birads_q_all.append(_bq[_bi2].numpy())
                        birads_base_pred_all.append(int(_base_bpred[_bi2]))
                        birads_base_prob_all.append(_base_bprob[_bi2].numpy())
                        birads_base_q_all.append(_base_bq[_bi2].numpy())

            # --- Pathology (benign/malignant) accumulation (added) ---
            if outputs.get('pathology_meas_logits') is not None and 'has_pathology' in batch and 'pathology_index' in batch:
                _hp = batch['has_pathology']
                _pidx_gt = batch['pathology_index']
                _valid_p = _hp & (_pidx_gt >= 0)
                if _valid_p.any():
                    _path_probs = torch.softmax(outputs['pathology_meas_logits'].float(), dim=1).cpu()
                    _path_pred = _path_probs.argmax(dim=1)
                    _base_path_logits = outputs.get('pathology_base_logits', None)
                    if _base_path_logits is not None:
                        _base_path_probs = torch.softmax(
                            _base_path_logits.float(), dim=1
                        ).cpu()
                        _base_path_pred = _base_path_probs.argmax(dim=1)
                    else:
                        _base_path_probs = _path_probs
                        _base_path_pred = _path_pred
                    for _pi2 in _valid_p.nonzero(as_tuple=True)[0].tolist():
                        pathology_pred_all.append(int(_path_pred[_pi2]))
                        pathology_gt_all.append(int(_pidx_gt[_pi2]))
                        pathology_prob_all.append(float(_path_probs[_pi2, 1]))  # P(malignant)
                        pathology_base_pred_all.append(int(_base_path_pred[_pi2]))
                        pathology_base_prob_all.append(float(_base_path_probs[_pi2, 1]))

            # --- Detection Loss (MRI only, OFFICIAL DINO ONLY) ---
            img_h, img_w = imgs.shape[-2], imgs.shape[-1]
            det_match_indices = None
            active_to_batch_idx = None

            det_batch_idx = outputs.get("det_batch_idx", None)

            if (
                outputs.get("detection", None) is not None
                and det_targets is not None
                and det_batch_idx is not None
                and det_batch_idx.numel() > 0
            ):
                det_active_mask = torch.zeros(imgs.shape[0], dtype=torch.bool, device=device)
                det_active_mask[det_batch_idx] = True
                active_to_batch_idx = det_batch_idx

                det_targets_active, _ = prepare_detection_targets(
                    det_targets,
                    det_active_mask,
                    imgs.shape[-2],
                    imgs.shape[-1],
                    num_classes=det_num_classes,
                    device=device,
                )

                if len(det_targets_active) == det_batch_idx.numel():
                    # outputs["detection"] is FULL-BATCH during validation/eval,
                    # so we slice back to active MRI detection samples here.
                    det_out = {
                        "pred_logits": outputs["detection"]["pred_logits"][det_batch_idx].float(),
                        "pred_boxes": outputs["detection"]["pred_boxes"][det_batch_idx].float(),
                    }

                    if "aux_outputs" in outputs["detection"] and outputs["detection"]["aux_outputs"] is not None:
                        det_out["aux_outputs"] = [
                            {
                                "pred_logits": aux["pred_logits"][det_batch_idx].float(),
                                "pred_boxes": aux["pred_boxes"][det_batch_idx].float(),
                            }
                            for aux in outputs["detection"]["aux_outputs"]
                        ]

                    if "dn_meta" in outputs["detection"]:
                        det_out["dn_meta"] = outputs["detection"]["dn_meta"]

                    with torch.cuda.amp.autocast(enabled=False):
                        det_loss_dict, indices_list = model.detector.compute_loss(
                            det_out,
                            det_targets_active,
                            return_indices=True,
                        )

                    det_match_indices = (
                        indices_list[-1]
                        if isinstance(indices_list, list) and len(indices_list) > 0
                        else None
                    )

                    l_det = torch.zeros((), device=device)
                    for k, v in det_loss_dict.items():
                        if k in model.detector.criterion.weight_dict:
                            l_det = l_det + model.detector.criterion.weight_dict[k] * v

                    if torch.isfinite(l_det):
                        batch_loss_sum += W_DET * l_det
                        losses["det_loss"].append(float(l_det.item()))
                        loss_terms.append(W_DET * l_det)
                        weight_terms.append(W_DET)
                    else:
                        print("DEBUG: NaN/Inf in validation Detection Loss!")
                else:
                    print(
                        f"DEBUG: validation detection target count mismatch: "
                        f"det_batch_idx={det_batch_idx.numel()} vs det_targets_active={len(det_targets_active)}"
                    )

                # --- Detection metrics ---
                pred_logits_all = outputs["detection"]["pred_logits"].float()
                pred_boxes_all = outputs["detection"]["pred_boxes"].float()

                for b_idx in det_batch_idx.tolist():
                    pred_logits_b = pred_logits_all[b_idx]
                    pred_boxes_b = pred_boxes_all[b_idx]

                    gt_boxes_b = det_targets[b_idx]["boxes"]
                    gt_labels_b = det_targets[b_idx]["labels"]

                    pred_boxes_abs, pred_labels_b, pred_scores_b, _ = model.postprocess_dino_predictions(
                        pred_logits_b,
                        pred_boxes_b,
                        img_w,
                        img_h,
                        score_thresh=0.1,
                        topk=20,
                        nms_iou=0.4,
                    )

                    matches = match_detection_predictions_to_gt_class_aware(
                        pred_boxes_abs,
                        pred_labels_b,
                        gt_boxes_b,
                        gt_labels_b,
                        iou_thresh=0.5,
                    )

                    matched_pred = set()
                    matched_gt = set()

                    for p_idx, g_idx, iou in matches:
                        matched_pred.add(p_idx)
                        matched_gt.add(g_idx)

                        all_det_tp += 1
                        all_det_ious.append(iou)
                        all_det_pred_labels.append(int(pred_labels_b[p_idx].item()))
                        all_det_gt_labels.append(int(gt_labels_b[g_idx].item()))
                        all_det_scores.append(float(pred_scores_b[p_idx].item()))

                    all_det_fp += max(0, pred_boxes_abs.shape[0] - len(matched_pred))
                    all_det_fn += max(0, gt_boxes_b.shape[0] - len(matched_gt))

            # --------------------------------------------------------------
            # Final batch loss logging
            # --------------------------------------------------------------
            if len(loss_terms) > 0:
                batch_loss = torch.stack(loss_terms).sum() / (sum(weight_terms) + 1e-8)
                losses['loss'].append(batch_loss.item())
            
            if val_rep:
                # ------------------------------------------------------------------
                # 2. Text Generation Validation (Guided by Predictions)
                # ------------------------------------------------------------------
                # We check if *any* sample needs reporting to enter the generation block
                if tokenizer and any('report_generation' in t for t in batch['tasks']):
                    original_padding_side = tokenizer.padding_side
                    tokenizer.padding_side = "left"
                    if (
                        max_val_samples is None
                        or len(all_preds_text) < max_val_samples
                    ):
                        
                        # 1. Get Predicted Tags from Class Logits (Same as Test Logic)
                        # Note: We need logits for ALL samples to generate prompts
                        batch_logits = outputs.get('class_logits')
                        batch_mask = batch.get('class_mask', None)
                        
                        if batch_logits is not None and batch_mask is not None:
                            m = batch_mask.to(device, non_blocking=True).bool()
                            masked_logits = batch_logits.clone()
                            masked_logits[~m] = -1e9
                            batch_probs = torch.sigmoid(masked_logits)
                            batch_preds = (batch_probs > 0.5).int().cpu().numpy()
                            class_names = loader.dataset.class_names
                            
                            # Identify indices that actually have reports
                            rep_indices = [i for i, t in enumerate(batch['tasks']) if 'report_generation' in t]
                            
                            for idx in rep_indices:
                                if (
                                    max_val_samples is not None
                                    and len(all_preds_text) >= max_val_samples
                                ):
                                    break
                                
                                ds_name_i = batch['ds_name'][idx]                  # per-sample dataset name
                                mod_id_i  = int(mod_ids[idx].item())               # per-sample modality id
                                mod_name  = id2mod.get(mod_id_i, f"Mod{mod_id_i}") # per-sample modality name
    
                                if not use_prompt_guidance:
                                    content_str = make_report_prompt_content(None, use_guidance=False)

                                elif ds_name_i == "LSS_MRI_AISSLab":
                                    tags = build_mri_predicted_tags_from_crop(
                                        outputs=outputs,
                                        sample_idx=idx,
                                        det_class_names=loader.dataset.det_class_names,
                                        severity_names=["Normal", "Stenosis", "Severe Stenosis"],
                                        skip_normal=True,
                                    )
                                    content_str = make_report_prompt_content(tags, use_guidance=True)

                                elif ds_name_i == "BUS_BRA":
                                    # Predicted-descriptor prompt (predicted mask → measurements,
                                    # pathology head, BI-RADS head) — mirrors test_multitask logic
                                    # Specialist mode -> GT descriptors (predicted heads untrained).
                                    if bool(getattr(model, "use_gt_descriptors", False)):
                                        _mvec = batch["measurements"][idx] if "measurements" in batch else []
                                        _bidx = int(batch["birads_index"][idx].item()) if "birads_index" in batch else -1
                                        _pidx = int(batch["pathology_index"][idx].item()) if "pathology_index" in batch else -1
                                        _full_gt_morph = (
                                            batch.get("report_morphology_gt", [{}] * len(batch["ds_name"]))[idx]
                                            if isinstance(batch.get("report_morphology_gt", None), list) else {}
                                        )
                                        _desc = loader.dataset.build_gt_descriptor_dict(
                                            _mvec, _bidx, _pidx, full_morphology=_full_gt_morph
                                        )
                                    else:
                                        content_str, _desc = build_busbra_predicted_prompt(
                                            outputs=outputs,
                                            sample_idx=idx,
                                            dataset=loader.dataset,
                                        )
                                    if bool(getattr(model, "use_gt_descriptors", False)):
                                        content_str = loader.dataset._build_descriptor_content(
                                            _desc, enforce_pixel_units=True
                                        )

                                else:
                                    active_indices = np.where(batch_preds[idx] == 1)[0]
                                    tags = [class_names[k] for k in active_indices]
                                    if NORMAL_LABEL in tags:
                                        tags.remove(NORMAL_LABEL)
                                    content_str = make_report_prompt_content(tags, use_guidance=True)

                                messages = [
                                    {"role": "system", "content": f"You are an expert {mod_name} radiologist. Describe the findings."},
                                    {"role": "user", "content": content_str}
                                ]
                                
                                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                                enc = tokenizer(prompt_text, return_tensors='pt', add_special_tokens=False).to(device, non_blocking=True)
                                
                                # B. Generate
                                with torch.amp.autocast('cuda'):
                                # Slice visual tokens for just this sample
                                    vis_feat = outputs['visual_tokens'][idx:idx+1]
                                
                                gen_ids = model.report_decoder.generate(
                                    visual_features=vis_feat, 
                                    prompt_ids=enc['input_ids'], 
                                    attention_mask=enc['attention_mask'], do_sample=False, num_beams=3
                                )
                                
                                # 2. Decode Directly (No Slicing)
                                pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                                # 3. Safety Cleanup (The "Previous" logic, but safer)
                                # Only split if the model accidentally repeated the header
                                if "Report:" in pred_text:
                                    pred_text = pred_text.split("Report:")[-1].strip()
                                elif "findings." in pred_text[:20].lower(): # Handle "Describe the findings." echo
                                    pred_text = pred_text.split("findings.")[-1].strip()
                                
                                # Clean Echo
                                if "Report:" in pred_text: pred_text = pred_text.split("Report:")[-1].strip()
                                elif "Describe the findings." in pred_text: pred_text = pred_text.split("Describe the findings.")[-1].strip()
                                
                                gt_text = batch['report_text'][idx]
                                
                                all_preds_text.append(pred_text)
                                all_refs_text.append(gt_text)
                                all_report_ds_names.append(ds_name_i)

                                # Sanity Check Print
                                if not first_batch_processed:
                                    print(f"\n--- Validation Sanity Check (Epoch {epoch}) ---")
                                    print(f"Prompt Used: {content_str}")
                                    print(f"  GT:   {gt_text}") 
                                    print(f"  PRED: {pred_text}")
                                    print("-" * 30)
                                    first_batch_processed = True
                                    
                    tokenizer.padding_side = original_padding_side
    # ------------------------------------------------------------------
    # 3. Final Metrics
    # ------------------------------------------------------------------
    metrics = {k: np.mean(v) if v else 0.0 for k, v in losses.items()}
    
    # ---------------------------------------------------------
    # NLP
    # ---------------------------------------------------------
    if len(all_preds_text) > 0:
        print(f"\nComputing NLP Metrics on {len(all_preds_text)} samples...")

        all_report_ds_names = np.array(all_report_ds_names)
        unique_datasets = np.unique(all_report_ds_names)

        for ds in unique_datasets:
            ds_idx = all_report_ds_names == ds
            preds_ds = [all_preds_text[i] for i in range(len(all_preds_text)) if ds_idx[i]]
            refs_ds = [all_refs_text[i] for i in range(len(all_refs_text)) if ds_idx[i]]

            if len(preds_ds) == 0:
                continue

            meteor_scores, rouge_scores = [], []

            for pred, ref in zip(preds_ds, refs_ds):
                pred_clean = normalize(pred)
                ref_clean = normalize(ref)

                pred_tokens = pred_clean.split()
                ref_tokens = ref_clean.split()

                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
                    rouge_scores.append(rouge_evaluator.score(ref, pred)['rougeL'].fmeasure)
                else:
                    meteor_scores.append(0.0)
                    rouge_scores.append(0.0)

            metrics[f'Val_{ds}_METEOR'] = np.mean(meteor_scores)
            metrics[f'Val_{ds}_ROUGE_L'] = np.mean(rouge_scores)

            print(f"\n  [Report: {ds}]")
            print(f"    METEOR  : {metrics[f'Val_{ds}_METEOR']:.4f}")
            print(f"    ROUGE-L : {metrics[f'Val_{ds}_ROUGE_L']:.4f}")
    
    # ---------------------------------------------------------
    # Classification metrics
    # ---------------------------------------------------------
    if len(all_targets_cls) > 0:
        y_true = np.concatenate(all_targets_cls, axis=0)
        y_probs = np.concatenate(all_preds_probs, axis=0)
        y_mask = np.concatenate(all_masks_cls, axis=0)
        ds_names = np.array(all_ds_names)
        C = y_true.shape[1]

        # Per-class threshold tuning on this val set.
        # A fixed 0.5 cutoff is a poor operating point for a negatively-biased
        # model. Sweep a grid and pick the threshold that maximises per-class F1.
        # AUC/mAP are threshold-free and are unaffected by this.
        val_thresholds = np.full(C, 0.5, dtype=np.float32)
        _thresh_grid = np.linspace(0.05, 0.90, 35)
        for _c in range(C):
            _m = y_mask[:, _c] == 1
            if _m.sum() < 10 or y_true[_m, _c].sum() == 0:
                continue   # too few positives to tune
            _f1s = [
                f1_score(y_true[_m, _c], (y_probs[_m, _c] > _t).astype(int), zero_division=0)
                for _t in _thresh_grid
            ]
            val_thresholds[_c] = _thresh_grid[int(np.argmax(_f1s))]
        y_pred = (y_probs > val_thresholds).astype(int)
        print(f"  [Thresh] per-class val-tuned: "
              f"min={val_thresholds.min():.2f}  max={val_thresholds.max():.2f}  "
              f"mean={val_thresholds.mean():.2f}")

        auc_list = []
        ap_list = []

        for c in range(C):
            valid = y_mask[:, c] == 1
            if valid.sum() > 5:
                y_c = y_true[valid, c]
                p_c = y_probs[valid, c]
                if len(np.unique(y_c)) > 1:
                    auc_list.append(roc_auc_score(y_c, p_c))
                    ap_list.append(average_precision_score(y_c, p_c))

        metrics['Val_Macro_AUC'] = np.mean(auc_list) if auc_list else 0.0
        metrics['Val_mAP'] = np.mean(ap_list) if ap_list else 0.0

        valid_all = y_mask == 1
        y_true_flat = y_true[valid_all]
        y_pred_flat = y_pred[valid_all]

        if len(y_true_flat) > 0:
            metrics['Val_Macro_F1'] = f1_score(y_true_flat, y_pred_flat, average='macro')
            metrics['Val_Micro_F1'] = f1_score(y_true_flat, y_pred_flat, average='micro')
            metrics['Val_Hamming'] = hamming_loss(y_true_flat, y_pred_flat)
        else:
            metrics['Val_Macro_F1'] = 0.0
            metrics['Val_Micro_F1'] = 0.0
            metrics['Val_Hamming'] = 0.0

        print(f"\n[Val Cls GLOBAL]")
        print(f"  Macro AUC : {metrics['Val_Macro_AUC']:.4f}")
        print(f"  mAP       : {metrics['Val_mAP']:.4f}")
        print(f"  Macro F1  : {metrics['Val_Macro_F1']:.4f}")
        print(f"  Micro F1  : {metrics['Val_Micro_F1']:.4f}")
        print(f"  Hamming   : {metrics['Val_Hamming']:.4f}")

        unique_datasets = np.unique(ds_names)
        for ds in unique_datasets:
            ds_idx = ds_names == ds
            y_true_ds = y_true[ds_idx]
            y_probs_ds = y_probs[ds_idx]
            y_mask_ds = y_mask[ds_idx]

            ds_ap = []
            ds_auc = []

            for c in range(C):
                valid = y_mask_ds[:, c] == 1
                if valid.sum() > 5:
                    y_c = y_true_ds[valid, c]
                    p_c = y_probs_ds[valid, c]
                    if len(np.unique(y_c)) > 1:
                        ds_ap.append(average_precision_score(y_c, p_c))
                        ds_auc.append(roc_auc_score(y_c, p_c))

            if ds_ap:
                metrics[f'Val_{ds}_mAP'] = np.mean(ds_ap)
                metrics[f'Val_{ds}_AUC'] = np.mean(ds_auc)

                print(f"\n  [{ds}]")
                print(f"    mAP : {metrics[f'Val_{ds}_mAP']:.4f}")
                print(f"    AUC : {metrics[f'Val_{ds}_AUC']:.4f}")

        try:
            unique_datasets = np.unique(ds_names)
            for ds in unique_datasets:
                ds_idx = ds_names == ds
                y_true_ds = y_true[ds_idx]
                y_pred_ds = y_pred[ds_idx]
                y_mask_ds = y_mask[ds_idx]

                print(f"\n📊 Confusion Matrices for Dataset: {ds}")

                for c, class_name in enumerate(class_names):
                    valid = y_mask_ds[:, c] == 1
                    if valid.sum() < 5:
                        continue

                    y_c = y_true_ds[valid, c]
                    y_p = y_pred_ds[valid, c]

                    if len(np.unique(y_c)) < 2:
                        continue

                    cm = confusion_matrix(y_c, y_p, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()

                    print(f"\n  Class: {class_name}")
                    print(f"    TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")

                    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
                    sensitivity = tp / (tp + fn + 1e-8)
                    specificity = tn / (tn + fp + 1e-8)

                    print(f"    Acc: {acc:.4f}")
                    print(f"    Sensitivity: {sensitivity:.4f}")
                    print(f"    Specificity: {specificity:.4f}")
        except Exception as e:
            print(f"Error calculating Acc: {e}")
        
    # ---------------------------------------------------------
    # Detection metrics
    # ---------------------------------------------------------
    det_precision = all_det_tp / (all_det_tp + all_det_fp + 1e-8)
    det_recall = all_det_tp / (all_det_tp + all_det_fn + 1e-8)
    det_f1 = 2.0 * det_precision * det_recall / (det_precision + det_recall + 1e-8)

    metrics['Val_Det_Precision'] = det_precision
    metrics['Val_Det_Recall'] = det_recall
    metrics['Val_Det_F1'] = det_f1
    metrics['Val_Det_MeanIoU'] = float(np.mean(all_det_ious)) if len(all_det_ious) > 0 else 0.0

    if len(all_det_gt_labels) > 0:
        metrics['Val_Det_Label_Accuracy'] = accuracy_score(all_det_gt_labels, all_det_pred_labels)
        metrics['Val_Det_Label_F1_Macro'] = f1_score(all_det_gt_labels, all_det_pred_labels, average='macro')
    else:
        metrics['Val_Det_Label_Accuracy'] = 0.0
        metrics['Val_Det_Label_F1_Macro'] = 0.0

    print(f"\n[Val Detection]")
    print(f"  Precision      : {metrics['Val_Det_Precision']:.4f}")
    print(f"  Recall         : {metrics['Val_Det_Recall']:.4f}")
    print(f"  F1             : {metrics['Val_Det_F1']:.4f}")
    print(f"  Mean IoU       : {metrics['Val_Det_MeanIoU']:.4f}")
    print(f"  Label Accuracy : {metrics['Val_Det_Label_Accuracy']:.4f}")
    print(f"  Label F1 Macro : {metrics['Val_Det_Label_F1_Macro']:.4f}")
    
    # Multiclass metrics
    for ds, store in multiclass_val_data.items():
        if len(store["targets"]) == 0:
            continue

        y_true = np.array(store["targets"], dtype=np.int64)
        y_pred = np.array(store["preds"], dtype=np.int64)
        y_prob = np.array(store["probs"], dtype=np.float32)

        num_classes = y_prob.shape[1]

        # For LSS_MRI_ROI_CLS, these are your local classes
        if ds == "LSS_MRI_ROI_CLS":
            mc_class_names = ["Normal", "Stenosis", "Severe Stenosis"]
        else:
            mc_class_names = [f"Class {i}" for i in range(num_classes)]

        metrics[f'Val_{ds}_Acc'] = accuracy_score(y_true, y_pred)

        # Precision
        metrics[f'Val_{ds}_MacroPrecision'] = precision_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        metrics[f'Val_{ds}_WeightedPrecision'] = precision_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )

        # Recall
        metrics[f'Val_{ds}_MacroRecall'] = recall_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        metrics[f'Val_{ds}_WeightedRecall'] = recall_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )

        # F1
        metrics[f'Val_{ds}_MacroF1'] = f1_score(
            y_true,
            y_pred,
            average='macro',
            zero_division=0
        )

        metrics[f'Val_{ds}_WeightedF1'] = f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0
        )
        # -----------------------------
        # Multiclass mAP
        # -----------------------------
        mc_map, class_ap = compute_multiclass_map(
            y_true,
            y_prob,
            num_classes=num_classes
        )
        metrics[f'Val_{ds}_mAP'] = mc_map

        # -----------------------------
        # Multiclass ROC-AUC + ROC curve
        # -----------------------------
        try:
            present_classes = np.unique(y_true)

            if len(present_classes) >= 2:
                roc_dir = os.path.join(manager.run_dir, "roc_curves") if manager is not None else "roc_curves"
                save_path = os.path.join(
                    roc_dir,
                    f"Val_{ds}_ROC_epoch_{epoch}.png"
                )

                roc_info = plot_multiclass_roc_curve(
                    y_true=y_true,
                    y_prob=y_prob,
                    class_names=mc_class_names,
                    save_path=save_path,
                    title=f"Validation ROC Curve - {ds}"
                )

                metrics[f'Val_{ds}_MacroAUC'] = roc_info["macro_auc"]
                metrics[f'Val_{ds}_WeightedAUC'] = roc_info["weighted_auc"]
                metrics[f'Val_{ds}_MicroAUC'] = roc_info["micro_auc"]
                metrics[f'Val_{ds}_ClassAUC'] = roc_info["class_auc"]

                print(f"  ROC curve saved to: {save_path}")

            else:
                metrics[f'Val_{ds}_MacroAUC'] = 0.0
                metrics[f'Val_{ds}_WeightedAUC'] = 0.0
                metrics[f'Val_{ds}_MicroAUC'] = 0.0
                metrics[f'Val_{ds}_ClassAUC'] = []
                print(f"  Could not compute ROC-AUC for {ds}: only one class present.")

        except Exception as e:
            print(f"  Could not compute/save ROC curve for {ds}: {e}")
            metrics[f'Val_{ds}_MacroAUC'] = 0.0
            metrics[f'Val_{ds}_WeightedAUC'] = 0.0
            metrics[f'Val_{ds}_MicroAUC'] = 0.0
            metrics[f'Val_{ds}_ClassAUC'] = []

        print(f"\n[Val {ds} Multiclass]")
        print(f"  Accuracy           : {metrics[f'Val_{ds}_Acc']:.4f}")
        print(f"  Macro Precision    : {metrics[f'Val_{ds}_MacroPrecision']:.4f}")
        print(f"  Weighted Precision : {metrics[f'Val_{ds}_WeightedPrecision']:.4f}")
        print(f"  Macro Recall       : {metrics[f'Val_{ds}_MacroRecall']:.4f}")
        print(f"  Weighted Recall    : {metrics[f'Val_{ds}_WeightedRecall']:.4f}")
        print(f"  Macro F1           : {metrics[f'Val_{ds}_MacroF1']:.4f}")
        print(f"  Weighted F1        : {metrics[f'Val_{ds}_WeightedF1']:.4f}")
        print(f"  mAP                : {metrics[f'Val_{ds}_mAP']:.4f}")
        print(f"  Macro AUC          : {metrics[f'Val_{ds}_MacroAUC']:.4f}")
        print(f"  Weighted AUC       : {metrics[f'Val_{ds}_WeightedAUC']:.4f}")
        print(f"  Micro AUC          : {metrics[f'Val_{ds}_MicroAUC']:.4f}")
        if metrics[f'Val_{ds}_ClassAUC']:
            print("  Class AUC    :")
            for name, auc_value in zip(mc_class_names, metrics[f'Val_{ds}_ClassAUC']):
                if np.isnan(auc_value):
                    print(f"    {name}: NaN")
                else:
                    print(f"    {name}: {auc_value:.4f}")

        try:
            cm = confusion_matrix(y_true, y_pred)
            print("  Confusion Matrix:")
            print(cm)
        except Exception as e:
            print(f"  Could not compute multiclass confusion matrix for {ds}: {e}")

                
    # ---- per-dataset segmentation Dice summary (added) ----
    if seg_dice_by_ds:
        print("\n  ===== Per-dataset Segmentation Dice (val) =====")
        for _ds in sorted(seg_dice_by_ds.keys()):
            _all = seg_dice_by_ds[_ds]
            _les = seg_dice_lesion_by_ds.get(_ds, [])
            _mean_all = float(np.mean(_all)) if _all else float("nan")
            metrics["Val_%s_SegDice" % _ds] = _mean_all
            if _les:
                _mean_les = float(np.mean(_les))
                metrics["Val_%s_SegDice_lesion" % _ds] = _mean_les
                print("  %-22s Dice(all)=%.4f  Dice(lesion-present)=%.4f  (n=%d, lesion=%d)"
                      % (_ds, _mean_all, _mean_les, len(_all), len(_les)))
            else:
                print("  %-22s Dice(all)=%.4f  (n=%d)" % (_ds, _mean_all, len(_all)))
            _base_vals = seg_base_dice_by_ds.get(_ds, [])
            if len(_base_vals) == len(_all) and _base_vals:
                _base_mean = float(np.mean(_base_vals))
                metrics[f"Val_{_ds}_SegDice_Base"] = _base_mean
                metrics[f"Val_{_ds}_SegDice_GradifGain"] = _mean_all - _base_mean
                print("    GraDiF gain vs identity Dice=%+.4f" % (_mean_all - _base_mean))

    # Fit cumulative decision thresholds on validation.  The calibrated values
    # are stored in the model state and therefore travel with best_model.pth.
    if birads_gt_all and len(birads_q_all) == len(birads_gt_all):
        _core = model.module if hasattr(model, 'module') else model
        _old_thr = getattr(_core, 'birads_decode_thresholds', None)
        _old_np = _old_thr.detach().cpu().numpy() if _old_thr is not None else None
        _cal_thr = fit_birads_decode_thresholds(
            birads_q_all, birads_gt_all, _old_np
        )
        if _old_thr is not None:
            _old_thr.copy_(torch.as_tensor(
                _cal_thr, device=_old_thr.device, dtype=_old_thr.dtype
            ))
        _q_arr = np.asarray(birads_q_all)
        birads_pred_all = (
            _q_arr > _cal_thr.reshape(1, -1)
        ).sum(axis=1).astype(int).tolist()
        if len(birads_base_q_all) == len(birads_gt_all):
            _bq_arr = np.asarray(birads_base_q_all)
            birads_base_pred_all = (
                _bq_arr > _cal_thr.reshape(1, -1)
            ).sum(axis=1).astype(int).tolist()
        metrics["Val_BIRADS_Threshold_0"] = float(_cal_thr[0])
        metrics["Val_BIRADS_Threshold_1"] = float(_cal_thr[1])
        metrics["Val_BIRADS_Threshold_2"] = float(_cal_thr[2])
        print("  [BI-RADS calibration] thresholds=" + ", ".join(
            f"{v:.3f}" for v in _cal_thr
        ))

    # ---- BI-RADS metrics: multiclass + ordinal paper-standard set ----
    if birads_gt_all:
        _bnames = getattr(loader.dataset, 'birads_class_names', None) or ["2", "3", "4", "5"]
        _bmetrics, _bcm, _bperclass = compute_birads_metrics(
            birads_gt_all, birads_pred_all, birads_prob_all,
            class_names=_bnames, prefix="Val_BIRADS_",
        )
        metrics.update(_bmetrics)

        print("\n[Val BI-RADS]  (n=%d)" % len(birads_gt_all))
        print("  Exact Acc       : %.4f" % metrics['Val_BIRADS_Acc'])
        print("  Balanced Acc    : %.4f" % metrics['Val_BIRADS_BalancedAcc'])
        print("  Macro Precision : %.4f" % metrics['Val_BIRADS_MacroPrecision'])
        print("  Macro Recall    : %.4f" % metrics['Val_BIRADS_MacroRecall'])
        print("  Macro F1        : %.4f" % metrics['Val_BIRADS_MacroF1'])
        print("  Macro AUC (OvR) : %.4f" % metrics.get('Val_BIRADS_MacroAUC_OvR', float('nan')))
        print("  MAE             : %.4f" % metrics['Val_BIRADS_MAE'])
        print("  Within-1 Acc    : %.4f" % metrics['Val_BIRADS_Within1'])
        print("  Quadratic Kappa : %.4f" % metrics['Val_BIRADS_QWK'])
        for _row in _bperclass:
            print(
                "  BI-RADS %-3s: P=%.4f R=%.4f F1=%.4f (n=%d)"
                % (_row['name'], _row['precision'], _row['recall'], _row['f1'], _row['support'])
            )
        print("  Confusion matrix [rows=GT, cols=Pred]:")
        print(_bcm)

        if len(birads_base_pred_all) == len(birads_gt_all):
            _base_metrics, _, _ = compute_birads_metrics(
                birads_gt_all,
                birads_base_pred_all,
                birads_base_prob_all,
                class_names=_bnames,
                prefix="Val_BIRADS_Base_",
            )
            metrics.update(_base_metrics)
            metrics["Val_BIRADS_GradifGain_MacroF1"] = (
                metrics["Val_BIRADS_MacroF1"]
                - metrics["Val_BIRADS_Base_MacroF1"]
            )
            metrics["Val_BIRADS_GradifGain_QWK"] = (
                metrics["Val_BIRADS_QWK"]
                - metrics["Val_BIRADS_Base_QWK"]
            )
            metrics["Val_BIRADS_GradifGain_MacroAUC"] = (
                metrics.get("Val_BIRADS_MacroAUC_OvR", 0.0)
                - metrics.get("Val_BIRADS_Base_MacroAUC_OvR", 0.0)
            )
            print(
                "  GraDiF gain vs identity: MacroF1=%+.4f QWK=%+.4f AUC=%+.4f"
                % (
                    metrics["Val_BIRADS_GradifGain_MacroF1"],
                    metrics["Val_BIRADS_GradifGain_QWK"],
                    metrics["Val_BIRADS_GradifGain_MacroAUC"],
                )
            )

    # ---- Pathology (benign/malignant) metrics (added) ----
    if pathology_gt_all:
        _pp = np.array(pathology_pred_all)
        _pg = np.array(pathology_gt_all)
        _pprob = np.array(pathology_prob_all)
        _path_names = getattr(loader.dataset, 'pathology_class_names', None) or ['benign', 'malignant']

        metrics['Val_Pathology_Acc'] = float((_pp == _pg).mean())
        metrics['Val_Pathology_MacroF1'] = float(f1_score(_pg, _pp, average='macro', zero_division=0))
        metrics['Val_Pathology_WeightedF1'] = float(f1_score(_pg, _pp, average='weighted', zero_division=0))

        print("\n[Val Pathology]  (n=%d)" % len(_pg))
        print("  Accuracy      : %.4f" % metrics['Val_Pathology_Acc'])
        print("  Macro F1      : %.4f" % metrics['Val_Pathology_MacroF1'])
        print("  Weighted F1   : %.4f" % metrics['Val_Pathology_WeightedF1'])

        for _c in sorted(set(_pg.tolist())):
            _m = (_pg == _c)
            _cat = _path_names[_c] if _c < len(_path_names) else str(_c)
            _acc_c = float((_pp[_m] == _pg[_m]).mean())
            metrics['Val_Pathology_cat%s_Acc' % _cat] = _acc_c
            print("  Pathology %-10s : acc=%.4f (n=%d)" % (_cat, _acc_c, int(_m.sum())))

        # Binary-specific: sensitivity (malignant recall), specificity (benign recall), AUC
        if set(_pg.tolist()) <= {0, 1}:
            _pos = (_pg == 1)
            _neg = (_pg == 0)
            if _pos.any():
                metrics['Val_Pathology_Sensitivity'] = float((_pp[_pos] == 1).mean())
                print("  Sensitivity (malignant recall) : %.4f (n=%d)" % (metrics['Val_Pathology_Sensitivity'], int(_pos.sum())))
            if _neg.any():
                metrics['Val_Pathology_Specificity'] = float((_pp[_neg] == 0).mean())
                print("  Specificity (benign recall)    : %.4f (n=%d)" % (metrics['Val_Pathology_Specificity'], int(_neg.sum())))
            if _pos.any() and _neg.any():
                try:
                    metrics['Val_Pathology_AUC'] = float(roc_auc_score(_pg, _pprob))
                    print("  AUC                             : %.4f" % metrics['Val_Pathology_AUC'])
                except Exception as e:
                    print(f"  Could not compute Pathology AUC: {e}")

        if len(pathology_base_pred_all) == len(pathology_gt_all):
            _pbp = np.array(pathology_base_pred_all)
            _pbprob = np.array(pathology_base_prob_all)
            metrics['Val_Pathology_Base_MacroF1'] = float(
                f1_score(_pg, _pbp, average='macro', zero_division=0)
            )
            try:
                metrics['Val_Pathology_Base_AUC'] = float(
                    roc_auc_score(_pg, _pbprob)
                    if len(np.unique(_pg)) > 1 else 0.0
                )
            except Exception:
                metrics['Val_Pathology_Base_AUC'] = 0.0
            metrics['Val_Pathology_GradifGain_MacroF1'] = (
                metrics['Val_Pathology_MacroF1']
                - metrics['Val_Pathology_Base_MacroF1']
            )
            metrics['Val_Pathology_GradifGain_AUC'] = (
                metrics.get('Val_Pathology_AUC', 0.0)
                - metrics['Val_Pathology_Base_AUC']
            )
            print(
                "  GraDiF gain vs identity: MacroF1=%+.4f AUC=%+.4f"
                % (
                    metrics['Val_Pathology_GradifGain_MacroF1'],
                    metrics['Val_Pathology_GradifGain_AUC'],
                )
            )

        try:
            cm = confusion_matrix(_pg, _pp)
            print("  Confusion Matrix:")
            print(cm)
        except Exception as e:
            print(f"  Could not compute pathology confusion matrix for BUS_BRA: {e}")

    return metrics

class MoECapture:
    def __init__(self):
        self.outputs = {} # Store by expert index

    def make_hook(self, index):
        # This creates a unique hook for each expert
        def hook(module, input, output):
            self.outputs[index] = output.detach()
        return hook

def compute_multiclass_map(y_true, y_prob, num_classes=None):
    """
    Macro mAP for multiclass classification using one-vs-rest AP.

    Args:
        y_true: np.ndarray [N] integer class ids
        y_prob: np.ndarray [N, C] class probabilities
        num_classes: optional int

    Returns:
        macro_map: float
        per_class_ap: list of length C
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    if y_true.ndim != 1 or y_prob.ndim != 2:
        return 0.0, []

    N, C = y_prob.shape
    if num_classes is None:
        num_classes = C

    if N == 0 or C == 0:
        return 0.0, []

    # one-hot GT
    y_true_oh = np.zeros((N, num_classes), dtype=np.float32)
    valid = (y_true >= 0) & (y_true < num_classes)
    y_true_oh[np.arange(N)[valid], y_true[valid]] = 1.0

    ap_list = []
    for c in range(num_classes):
        y_c = y_true_oh[:, c]
        p_c = y_prob[:, c]

        # AP is only meaningful if class has both pos/neg examples
        if len(np.unique(y_c)) < 2:
            continue

        try:
            ap = average_precision_score(y_c, p_c)
            ap_list.append(float(ap))
        except Exception:
            continue

    macro_map = float(np.mean(ap_list)) if len(ap_list) > 0 else 0.0
    return macro_map, ap_list

@torch.no_grad()
def evaluate_detection_map50_from_storage(det_store, det_class_names):
    """
    Compute COCO-style mAP@50 from already collected test_multitask det_data[ds].

    det_store keys expected:
        pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except Exception as e:
        print(f"[DET mAP@50] pycocotools not available: {e}")
        return 0.0

    if det_class_names is None or len(det_class_names) == 0:
        print("[DET mAP@50] det_class_names missing.")
        return 0.0

    pred_boxes_list = det_store.get("pred_boxes", [])
    pred_labels_list = det_store.get("pred_labels", [])
    pred_scores_list = det_store.get("pred_scores", [])
    gt_boxes_list = det_store.get("gt_boxes", [])
    gt_labels_list = det_store.get("gt_labels", [])

    if len(gt_boxes_list) == 0:
        return 0.0

    categories = [{"id": i + 1, "name": name} for i, name in enumerate(det_class_names)]

    images_coco = []
    anns_coco = []
    results = []

    ann_id = 1

    for image_id, (gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores) in enumerate(
        zip(gt_boxes_list, gt_labels_list, pred_boxes_list, pred_labels_list, pred_scores_list),
        start=1
    ):
        gt_boxes = gt_boxes.detach().cpu()
        gt_labels = gt_labels.detach().cpu()
        pred_boxes = pred_boxes.detach().cpu()
        pred_labels = pred_labels.detach().cpu()
        pred_scores = pred_scores.detach().cpu()

        images_coco.append({
            "id": image_id,
            "width": 1,   # not used by COCOeval for bbox AP
            "height": 1,  # not used by COCOeval for bbox AP
            "file_name": f"img_{image_id}.png",
        })

        # GT annotations
        for j in range(gt_boxes.shape[0]):
            x1, y1, x2, y2 = gt_boxes[j].tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            anns_coco.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(gt_labels[j].item()) + 1,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

        # Predictions
        for s, lab, box in zip(pred_scores, pred_labels, pred_boxes):
            x1, y1, x2, y2 = box.tolist()
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)

            results.append({
                "image_id": image_id,
                "category_id": int(lab.item()) + 1,
                "bbox": [x1, y1, w, h],
                "score": float(s.item()),
            })

    if len(results) == 0:
        return 0.0

    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {"description": "test_multitask detection eval", "version": "1.0"},
        "licenses": [],
        "images": images_coco,
        "annotations": anns_coco,
        "categories": categories,
    }
    coco_gt.createIndex()

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()

    precision = coco_eval.eval.get("precision", None)
    if precision is None:
        return 0.0

    ious = np.array(coco_eval.params.iouThrs, dtype=np.float32)
    t_idx = np.where(np.isclose(ious, 0.50))[0]
    if len(t_idx) == 0:
        return 0.0

    t_idx = int(t_idx[0])
    a = 0
    m = len(coco_eval.params.maxDets) - 1

    p = precision[t_idx, :, :, a, m]
    p = p[p > -1]

    return float(np.mean(p)) if p.size > 0 else 0.0

def build_mri_predicted_tags_from_crop(det_level_labels, crop_sev_preds, det_class_names, severity_names=None, skip_normal=True):
    if severity_names is None:
        severity_names = ["Normal", "Stenosis", "Severe Stenosis"]

    tags = []
    seen = set()

    n = min(len(det_level_labels), len(crop_sev_preds))
    for i in range(n):
        level_id = int(det_level_labels[i])
        sev_id = int(crop_sev_preds[i])

        if skip_normal and sev_id == 0:
            continue

        level_name = det_class_names[level_id] if 0 <= level_id < len(det_class_names) else f"Level{level_id}"
        sev_name = severity_names[sev_id] if 0 <= sev_id < len(severity_names) else f"Severity{sev_id}"

        tag = f"{level_name} {sev_name}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags

def get_mri_severity_class_indices(class_names):
    """
    Find the indices of MRI ROI crop classes in the GLOBAL unified class schema.

    Expected canonical order:
        0 -> No_Finding
        1 -> Stenosis
        2 -> Severe Stenosis

    Backward-compatible aliases:
        No_Finding: ["no_finding", "no finding", "normal"]
        Stenosis: ["stenosis"]
        Severe Stenosis: ["severe stenosis", "severe_stenosis"]

    Returns:
        idxs  : list[int]   -> column indices in global class_names
        names : list[str]   -> resolved names from global class_names
    """
    canon_groups = [
        ["no_finding", "no finding", "normal"],
        ["stenosis"],
        ["severe stenosis", "severe_stenosis"],
    ]

    lowered = [str(c).strip().lower() for c in class_names]
    idxs = []
    names = []

    for aliases in canon_groups:
        found = None
        for alias in aliases:
            if alias in lowered:
                found = lowered.index(alias)
                break

        if found is None:
            raise ValueError(
                f"Could not find any of {aliases} in global class_names: {class_names}"
            )

        idxs.append(found)
        names.append(class_names[found])

    return idxs, names

def extract_mri_roi_crops_from_batch(imgs, pred_boxes_list, out_size=(224, 224), expand_ratio=0.20):
    crop_tensors = []
    crop_meta = []

    B, C, H, W = imgs.shape
    out_h, out_w = out_size

    for b in range(B):
        boxes_b = pred_boxes_list[b]
        if boxes_b is None or boxes_b.numel() == 0:
            continue

        for j in range(boxes_b.shape[0]):
            x1, y1, x2, y2 = boxes_b[j].tolist()

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            pad_x = int(round(bw * expand_ratio))
            pad_y = int(round(bh * expand_ratio))

            x1 = max(0, int(round(x1)) - pad_x)
            y1 = max(0, int(round(y1)) - pad_y)
            x2 = min(W, int(round(x2)) + pad_x)
            y2 = min(H, int(round(y2)) + pad_y)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = imgs[b:b+1, :, y1:y2, x1:x2]
            if crop.shape[-1] < 2 or crop.shape[-2] < 2:
                continue

            crop = F.interpolate(crop, size=(out_h, out_w), mode="bilinear", align_corners=False)
            crop_tensors.append(crop.squeeze(0))
            crop_meta.append((b, j))

    if len(crop_tensors) == 0:
        return None, []

    return torch.stack(crop_tensors, dim=0), crop_meta

def build_mri_predicted_tags_from_crop(
    pred_level_labels_b,
    pred_severity_ids_b,
    det_class_names,
    severity_names=None,
    skip_normal=True,
):
    """
    Build MRI report tags from:
      - detector level predictions
      - crop-classifier severity predictions

    Args:
        pred_level_labels_b: list[int] or tensor-like
            Detector level ids, e.g. [0, 3] -> ["L1-L2", "L4-L5"]

        pred_severity_ids_b: list[int] or tensor-like
            Crop-classifier severity ids in the 3-class scheme:
                0 -> No_Finding
                1 -> Stenosis
                2 -> Severe Stenosis

        det_class_names: list[str]
            Detection class names, e.g. ["L1-L2", ..., "L5-S1"]

        severity_names: list[str], optional
            Defaults to ["No_Finding", "Stenosis", "Severe Stenosis"]

        skip_normal: bool
            If True, No_Finding tags are omitted from report prompts.

    Returns:
        list[str]
            Example:
                ["L2-L3 Stenosis", "L4-L5 Severe Stenosis"]
    """
    if severity_names is None:
        severity_names = ["No_Finding", "Stenosis", "Severe Stenosis"]

    tags = []
    seen = set()

    n = min(len(pred_level_labels_b), len(pred_severity_ids_b))
    for k in range(n):
        level_id = int(pred_level_labels_b[k])
        sev_id = int(pred_severity_ids_b[k])

        if sev_id < 0 or sev_id >= len(severity_names):
            continue

        sev_name = severity_names[sev_id]

        if skip_normal and sev_name in ["No_Finding", "Normal"]:
            continue

        level_name = (
            det_class_names[level_id]
            if 0 <= level_id < len(det_class_names)
            else f"Level{level_id}"
        )

        tag = f"{level_name} {sev_name}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)

    return tags

# -----------------------------------------------------------------------------
# Test-time MoFX routing diagnostics
# -----------------------------------------------------------------------------
_MOFX_CONTEXT_ORDER = [
    "X-ray classification",
    "X-ray segmentation",
    "US segmentation",
    "US pathology",
    "US BI-RADS",
    "CT classification",
    "MRI segmentation",
    "MRI detection",
    "MRI severity",
    "Report generation",
]


def _unwrap_mofx_model(model):
    """Return the underlying multitask model for DP/DDP/torch.compile wrappers."""
    core = model.module if hasattr(model, "module") else model
    core = getattr(core, "_orig_mod", core)
    return core


def _safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.")
    return value or "unknown"


def _canonical_modality_name(name):
    text = str(name).strip()
    low = text.lower().replace("_", " ").replace("-", " ")
    if any(k in low for k in ("xray", "x ray", "cxr", "chest")):
        return "X-ray"
    if any(k in low for k in ("ultrasound", "ultra sound", "us", "busi", "bus bra", "busbra")):
        return "US"
    if "ct" in low or "hemorrhage" in low:
        return "CT"
    if any(k in low for k in ("mri", "magnetic", "lss")):
        return "MRI"
    return text or "Unknown"


def _mofx_context_name(stage, route, modality_name):
    """Map the actual stage/route/modality tuple to the paper-facing context."""
    stage = str(stage)
    route = str(route)
    modality = _canonical_modality_name(modality_name)

    if stage in {"c2", "c3", "c4_spatial"}:
        if route in {"0", "seg", "segmentation"}:
            if modality == "X-ray":
                return "X-ray segmentation"
            if modality == "US":
                return "US segmentation"
            if modality == "MRI":
                return "MRI segmentation"
            return f"{modality} segmentation"
        if route in {"3", "det", "detection"}:
            if modality == "MRI":
                return "MRI detection"
            return f"{modality} detection"

    if stage in {"c4", "c4_semantic"}:
        if route == "birads":
            return "US BI-RADS"
        if route == "pathology":
            return "US pathology"
        if route in {"report", "2", "reporting"}:
            return "Report generation"
        if route in {"classification", "1", "cls"}:
            if modality == "X-ray":
                return "X-ray classification"
            if modality == "CT":
                return "CT classification"
            if modality == "MRI":
                return "MRI severity"
            return f"{modality} classification"

    return f"{modality} | {stage} | {route}"


def _extract_test_routing_records(routing_accumulator, hierarchy, modality_id_to_name):
    """Convert accumulated tensors into one row per sample/expert."""
    records = []
    route_records = []
    global_sample_id = 0

    for stage_name, bank in hierarchy.banks.items():
        stage_store = routing_accumulator.stages.get(str(stage_name), None)
        if stage_store is None:
            continue

        expert_names = list(bank.expert_names)
        for route_name, store in stage_store["tasks"].items():
            if not store["weights"]:
                continue

            weights = torch.cat(store["weights"], dim=0).float()
            dense = (
                torch.cat(store["dense"], dim=0).float()
                if store["dense"] else weights.clone()
            )
            topk = (
                torch.cat(store["topk"], dim=0).long()
                if store["topk"] else None
            )
            modalities = (
                torch.cat(store["modalities"], dim=0).long().flatten()
                if store["modalities"] else torch.full(
                    (weights.shape[0],), -1, dtype=torch.long
                )
            )

            n = min(weights.shape[0], dense.shape[0], modalities.shape[0])
            if topk is not None:
                n = min(n, topk.shape[0])
            weights = weights[:n]
            dense = dense[:n]
            modalities = modalities[:n]
            if topk is not None:
                topk = topk[:n]

            entropy_natural = -(
                weights.clamp_min(1e-12)
                * weights.clamp_min(1e-12).log()
            ).sum(dim=1)
            top1 = weights.argmax(dim=1)
            top1_weight = weights.max(dim=1).values

            for sample_idx in range(n):
                mid = int(modalities[sample_idx].item())
                raw_modality = modality_id_to_name.get(mid, f"modality_{mid}")
                modality = _canonical_modality_name(raw_modality)
                context = _mofx_context_name(
                    stage_name, route_name, modality
                )
                chosen = set(
                    topk[sample_idx].tolist() if topk is not None else []
                )
                pair = "+".join(
                    expert_names[int(e)] for e in sorted(chosen)
                    if 0 <= int(e) < len(expert_names)
                )

                route_records.append({
                    "SampleID": int(global_sample_id),
                    "Stage": str(stage_name),
                    "Route": str(route_name),
                    "ModalityID": mid,
                    "Modality": modality,
                    "Context": context,
                    "NormalizedEntropy": float(
                        entropy_natural[sample_idx].item()
                        / math.log(max(len(expert_names), 2))
                    ),
                    "EffectiveExperts": float(
                        torch.exp(entropy_natural[sample_idx]).item()
                    ),
                    "Top1Expert": expert_names[int(top1[sample_idx])],
                    "Top1Weight": float(top1_weight[sample_idx].item()),
                    "TopKPair": pair,
                })

                for expert_idx, expert_name in enumerate(expert_names):
                    records.append({
                        "SampleID": int(global_sample_id),
                        "Stage": str(stage_name),
                        "Route": str(route_name),
                        "ModalityID": mid,
                        "Modality": modality,
                        "Context": context,
                        "ExpertID": int(expert_idx),
                        "Expert": str(expert_name),
                        "Weight": float(weights[sample_idx, expert_idx].item()),
                        "DenseProbability": float(dense[sample_idx, expert_idx].item()),
                        "Top1": int(expert_idx == int(top1[sample_idx])),
                        "SelectedTopK": int(expert_idx in chosen),
                    })
                global_sample_id += 1

    return pd.DataFrame(records), pd.DataFrame(route_records)


def _weighted_context_matrix(raw_df, value_column, expert_names):
    if raw_df.empty:
        return pd.DataFrame(index=_MOFX_CONTEXT_ORDER, columns=expert_names)
    matrix = raw_df.pivot_table(
        index="Context", columns="Expert", values=value_column,
        aggfunc="mean"
    )
    extra = [c for c in matrix.index if c not in _MOFX_CONTEXT_ORDER]
    order = _MOFX_CONTEXT_ORDER + sorted(extra)
    return matrix.reindex(index=order, columns=expert_names)


def _selection_share_matrix(raw_df, expert_names, selected_column):
    """Return expert share among selected assignments, summing to one per context."""
    if raw_df.empty:
        return pd.DataFrame(index=_MOFX_CONTEXT_ORDER, columns=expert_names)
    selected = raw_df.groupby(["Context", "Expert"])[selected_column].sum()
    selected = selected.unstack("Expert").reindex(columns=expert_names).fillna(0.0)
    denom = selected.sum(axis=1).replace(0.0, np.nan)
    selected = selected.div(denom, axis=0)
    extra = [c for c in selected.index if c not in _MOFX_CONTEXT_ORDER]
    return selected.reindex(_MOFX_CONTEXT_ORDER + sorted(extra))


def _plot_routing_heatmap(matrix, path, title, cbar_label, fmt=".3f"):
    if matrix.empty or matrix.dropna(how="all").empty:
        return
    height = max(5.0, 0.48 * len(matrix.index) + 2.0)
    width = max(8.0, 1.25 * len(matrix.columns) + 3.0)
    plt.figure(figsize=(width, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": cbar_label},
    )
    plt.title(title)
    plt.xlabel("Expert")
    plt.ylabel("Task-modality context")
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _build_router_health_tables(raw_df, route_df, routing_mode, expert_names):
    context_rows = []
    if not route_df.empty:
        for context, group in route_df.groupby("Context", sort=False):
            weight_group = raw_df[raw_df["Context"] == context]
            sample_weight_matrix = weight_group.pivot_table(
                index="SampleID", columns="Expert", values="Weight",
                aggfunc="first"
            ).reindex(columns=expert_names).fillna(0.0)
            mean_weights = sample_weight_matrix.mean(axis=0)
            routing_std = float(sample_weight_matrix.std(axis=0, ddof=0).mean())
            top1_usage = group["Top1Expert"].value_counts(normalize=True)
            pair_usage = group["TopKPair"].value_counts(normalize=True)
            dominant = str(mean_weights.idxmax()) if len(mean_weights) else ""
            dominant_mass = float(mean_weights.max()) if len(mean_weights) else 0.0
            top1_peak = float(top1_usage.max()) if len(top1_usage) else 0.0
            pair_peak = float(pair_usage.max()) if len(pair_usage) else 0.0
            normalized_entropy = float(group["NormalizedEntropy"].mean())
            effective_experts = float(group["EffectiveExperts"].mean())
            aggregate_entropy = -float(np.sum(
                np.where(
                    mean_weights.to_numpy() > 0,
                    mean_weights.to_numpy()
                    * np.log(np.clip(mean_weights.to_numpy(), 1e-12, None)),
                    0.0,
                )
            ))
            aggregate_effective = float(np.exp(aggregate_entropy))
            dead = [
                str(name) for name, value in mean_weights.items()
                if float(value) < 1e-3
            ]
            static_router = bool(routing_std < (1e-4 if routing_mode == "dense_soft" else 5e-3))
            concentrated = bool(
                dominant_mass >= (0.85 if routing_mode == "dense_soft" else 0.80)
                or top1_peak >= 0.90
                or effective_experts <= 1.50
            )
            pair_collapsed = bool(
                routing_mode != "dense_soft" and pair_peak >= 0.97
            )
            collapsed = bool(concentrated and (static_router or pair_collapsed))
            if collapsed:
                status = "collapsed"
            elif static_router:
                status = "near-static"
            elif concentrated:
                status = "concentrated"
            else:
                status = "input-varying"

            row = {
                "Context": context,
                "N": int(len(group)),
                "RoutingMode": routing_mode,
                "NormalizedEntropy": normalized_entropy,
                "EffectiveExpertsPerSample": effective_experts,
                "AggregateEffectiveExperts": aggregate_effective,
                "RoutingStd": routing_std,
                "MeanTop1Weight": float(group["Top1Weight"].mean()),
                "DominantExpert": dominant,
                "DominantMeanWeight": dominant_mass,
                "Top1PeakUsage": top1_peak,
                "DominantTopKPair": str(pair_usage.index[0]) if len(pair_usage) else "",
                "TopKPairPeakUsage": pair_peak,
                "DeadExpertsMeanWeightLT1e-3": ", ".join(dead),
                "NearStatic": static_router,
                "Concentrated": concentrated,
                "Collapsed": collapsed,
                "Status": status,
            }
            for name in expert_names:
                row[f"MeanWeight::{name}"] = float(mean_weights.get(name, 0.0))
            context_rows.append(row)

    stage_route_rows = []
    if not route_df.empty:
        for (stage, route), group in route_df.groupby(["Stage", "Route"], sort=False):
            raw_group = raw_df[(raw_df["Stage"] == stage) & (raw_df["Route"] == route)]
            mat = raw_group.pivot_table(
                index="SampleID", columns="Expert", values="Weight",
                aggfunc="first"
            ).reindex(columns=expert_names).fillna(0.0)
            means = mat.mean(axis=0)
            top1_usage = group["Top1Expert"].value_counts(normalize=True)
            pair_usage = group["TopKPair"].value_counts(normalize=True)
            stage_route_rows.append({
                "Stage": stage,
                "Route": route,
                "N": int(len(group)),
                "NormalizedEntropy": float(group["NormalizedEntropy"].mean()),
                "EffectiveExpertsPerSample": float(group["EffectiveExperts"].mean()),
                "RoutingStd": float(mat.std(axis=0, ddof=0).mean()),
                "MeanTop1Weight": float(group["Top1Weight"].mean()),
                "DominantExpert": str(means.idxmax()),
                "DominantMeanWeight": float(means.max()),
                "Top1PeakUsage": float(top1_usage.max()) if len(top1_usage) else 0.0,
                "DominantTopKPair": str(pair_usage.index[0]) if len(pair_usage) else "",
                "TopKPairPeakUsage": float(pair_usage.max()) if len(pair_usage) else 0.0,
            })
    return pd.DataFrame(context_rows), pd.DataFrame(stage_route_rows)


def _plot_routing_histograms(raw_df, expert_names, output_dir, routing_mode):
    if raw_df.empty:
        return
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    bins = np.linspace(0.0, 1.0, 31)

    contexts = ["ALL"] + [
        c for c in _MOFX_CONTEXT_ORDER if c in set(raw_df["Context"])
    ]
    for context in contexts:
        subset = raw_df if context == "ALL" else raw_df[raw_df["Context"] == context]
        if subset.empty:
            continue
        ncols = min(3, max(1, len(expert_names)))
        nrows = int(math.ceil(len(expert_names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows))
        axes = np.atleast_1d(axes).reshape(-1)
        for idx, expert in enumerate(expert_names):
            values = subset.loc[subset["Expert"] == expert, "Weight"].to_numpy()
            axes[idx].hist(values, bins=bins)
            axes[idx].set_title(str(expert))
            axes[idx].set_xlabel("Applied routing weight")
            axes[idx].set_ylabel("Samples")
            axes[idx].set_xlim(0.0, 1.0)
            axes[idx].grid(alpha=0.25)
        for idx in range(len(expert_names), len(axes)):
            axes[idx].axis("off")
        fig.suptitle(
            f"MoFX routing-weight distributions — {context} ({routing_mode})"
        )
        fig.tight_layout()
        filename = "routing_weight_histograms_all.png" if context == "ALL" else (
            f"routing_weight_histograms_{_safe_filename(context)}.png"
        )
        fig.savefig(os.path.join(hist_dir, filename), dpi=300, bbox_inches="tight")
        plt.close(fig)


def _summarize_gradif_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    epoch = int(payload.get("epoch", -1))
    weighted_entropy = 0.0
    weighted_effective = 0.0
    total_n = 0
    expert_sum = None
    expert_names = None

    for stage in payload.get("stages", {}).values():
        stage_names = stage.get("experts", None)
        if stage_names is None and isinstance(stage.get("stages"), dict):
            stage_names = stage.get("experts", None)
        if stage_names and expert_names is None:
            expert_names = list(stage_names)
        for task in stage.get("tasks", {}).values():
            n = int(task.get("n", 0))
            weights = task.get("mean_weights", [])
            if n <= 0 or not weights:
                continue
            arr = np.asarray(weights, dtype=float)
            if expert_sum is None:
                expert_sum = np.zeros_like(arr)
            if arr.shape != expert_sum.shape:
                continue
            weighted_entropy += n * float(task.get("normalized_entropy", np.nan))
            weighted_effective += n * float(task.get("effective_experts", np.nan))
            expert_sum += n * arr
            total_n += n

    row = {
        "Epoch": epoch,
        "NormalizedEntropy": weighted_entropy / total_n if total_n else np.nan,
        "EffectiveExperts": weighted_effective / total_n if total_n else np.nan,
        "NRouteSamples": int(total_n),
    }
    if expert_sum is not None and total_n:
        mean_weights = expert_sum / total_n
        for idx, value in enumerate(mean_weights):
            name = expert_names[idx] if expert_names and idx < len(expert_names) else f"expert_{idx}"
            row[f"Weight::{name}"] = float(value)
    return row


def _current_summary_epoch_row(summary, epoch, expert_names):
    weighted_entropy = 0.0
    weighted_effective = 0.0
    total_n = 0
    expert_sum = np.zeros(len(expert_names), dtype=float)
    for stage in summary.get("stages", {}).values():
        for task in stage.get("tasks", {}).values():
            n = int(task.get("n", 0))
            weights = np.asarray(task.get("mean_weights", []), dtype=float)
            if n <= 0 or weights.shape[0] != len(expert_names):
                continue
            weighted_entropy += n * float(task.get("normalized_entropy", np.nan))
            weighted_effective += n * float(task.get("effective_experts", np.nan))
            expert_sum += n * weights
            total_n += n
    row = {
        "Epoch": int(epoch),
        "NormalizedEntropy": weighted_entropy / total_n if total_n else np.nan,
        "EffectiveExperts": weighted_effective / total_n if total_n else np.nan,
        "NRouteSamples": int(total_n),
        "RoutingSplit": "test",
    }
    if total_n:
        for name, value in zip(expert_names, expert_sum / total_n):
            row[f"Weight::{name}"] = float(value)
    return row


def _build_routing_over_epochs(manager, current_summary, expert_names, output_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(manager.run_dir, "gradif_diagnostics_epoch_*.json"))):
        try:
            row = _summarize_gradif_json(path)
            row["RoutingSplit"] = "validation"
            rows.append(row)
        except Exception as exc:
            print(f"[MoFX diagnostics] Could not read {path}: {exc}")

    log_df = pd.DataFrame()
    best_epoch = -1
    if os.path.isfile(manager.log_file):
        try:
            log_df = pd.read_excel(manager.log_file)
            log_df["Epoch"] = pd.to_numeric(log_df.get("Epoch"), errors="coerce")
            log_df = log_df.dropna(subset=["Epoch"]).copy()
            log_df["Epoch"] = log_df["Epoch"].astype(int)
            if "Val_UnifiedScore" in log_df.columns:
                values = pd.to_numeric(log_df["Val_UnifiedScore"], errors="coerce")
                if values.notna().any():
                    best_epoch = int(log_df.loc[values.idxmax(), "Epoch"])
        except Exception as exc:
            print(f"[MoFX diagnostics] Could not read training log: {exc}")

    if best_epoch < 0:
        epochs = [int(row["Epoch"]) for row in rows if int(row["Epoch"]) >= 0]
        best_epoch = max(epochs) if epochs else 0

    # Keep the over-epoch curve validation-only whenever validation routing
    # diagnostics exist. Use the test checkpoint as a single fallback point
    # only when no epoch diagnostics were saved during training.
    if not rows:
        rows.append(_current_summary_epoch_row(
            current_summary, best_epoch, expert_names
        ))

    routing_df = pd.DataFrame(rows)
    if routing_df.empty:
        return routing_df
    routing_df = routing_df.sort_values(["Epoch", "RoutingSplit"]).reset_index(drop=True)

    if not log_df.empty:
        score_columns = [
            col for col in (
                "Val_UnifiedScore", "Val_UltrasoundScore", "Val_loss"
            ) if col in log_df.columns
        ]
        routing_df = routing_df.merge(
            log_df[["Epoch"] + score_columns].drop_duplicates("Epoch"),
            on="Epoch", how="left"
        )

    routing_df.to_csv(
        os.path.join(output_dir, "routing_over_epochs.csv"), index=False
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
    x = routing_df["Epoch"].to_numpy()

    if "Val_UnifiedScore" in routing_df.columns:
        axes[0].plot(x, routing_df["Val_UnifiedScore"], marker="o")
        axes[0].set_ylabel("Validation unified score")
    elif "Val_UltrasoundScore" in routing_df.columns:
        axes[0].plot(x, routing_df["Val_UltrasoundScore"], marker="o")
        axes[0].set_ylabel("Validation ultrasound score")
    elif "Val_loss" in routing_df.columns:
        axes[0].plot(x, routing_df["Val_loss"], marker="o")
        axes[0].set_ylabel("Validation loss")
    else:
        axes[0].text(0.5, 0.5, "No score column found", ha="center", va="center")
    axes[0].set_title("Unified performance over epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, routing_df["EffectiveExperts"], marker="o", label="Effective experts")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Effective experts")
    axes[1].set_title("Routing diversity over epochs")
    axes[1].grid(alpha=0.3)
    ax_entropy = axes[1].twinx()
    ax_entropy.plot(
        x, routing_df["NormalizedEntropy"], marker="x",
        linestyle="--", label="Normalized entropy"
    )
    ax_entropy.set_ylabel("Normalized entropy")
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = ax_entropy.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, loc="best")

    weight_columns = [c for c in routing_df.columns if c.startswith("Weight::")]
    for column in weight_columns:
        axes[2].plot(x, routing_df[column], marker="o", label=column.split("::", 1)[1])
    axes[2].set_title("Mean expert weights over epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Mean routing weight")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.3)
    if weight_columns:
        axes[2].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "routing_training_triptych.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    return routing_df



def test_multitask(
    model,
    loader,
    device,
    manager,
    tokenizer,
    pos_weight,
    *,
    analysis_tag="full",
    save_artifacts=True,
    append_to_log=True,
):
    print(f"\n--- Starting Granular Testing [{analysis_tag}] ---")
    debug_experts_output = False
    model.eval()

    cap = MoECapture()
    handles = []

    dataset_map = loader.dataset.modality_map
    mod_map = {v: k for k, v in dataset_map.items()}
    print(f"[Sanity Check] Using Modality Map: {mod_map}")

    results = defaultdict(lambda: defaultdict(list))

    det_data = defaultdict(lambda: {
        "pred_boxes": [],
        "pred_labels": [],
        "pred_scores": [],
        "gt_boxes": [],
        "gt_labels": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
    })

    # detection -> crop classifier evaluation on full-slice MRI
    mri_roi_crop_data = defaultdict(lambda: {
        "targets": [],
        "preds": [],
        "probs": [],
    })

    # report metrics
    text_data = defaultdict(lambda: {"preds": [], "gts": []})

    # classification metrics
    cls_data = defaultdict(lambda: {"probs": [], "targets": [], "masks": []})

    # BUS-BRA BI-RADS + pathology metrics (added)
    birads_pred_test, birads_gt_test, birads_prob_test = [], [], []
    birads_base_pred_test, birads_base_prob_test = [], []
    pathology_pred_test, pathology_gt_test, pathology_prob_test = [], [], []
    pathology_base_pred_test, pathology_base_prob_test = [], []

    report_export_data = defaultdict(list)

    rouge_evaluator = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    NORMAL_LABEL = "No_Finding"
    class_names = loader.dataset.class_names
    det_class_names = getattr(loader.dataset, "det_class_names", None)

    # IMPORTANT: this helper must now resolve 3-class MRI ROI columns:
    # ["No_Finding", "Stenosis", "Severe Stenosis"]
    severity_col_idx, severity_names_resolved = get_mri_severity_class_indices(class_names)

    task_names = ["Segmentation", "Classification", "Reporting", "Detection"]
    dataset_counters = defaultdict(int)

    # topk_dir = os.path.join(manager.run_dir, "topk_segmentation_examples")
    # topk_store = TopKSegmentationStore(k=10, save_dir=topk_dir)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Testing")):
            ds_names = batch["ds_name"]
            ds_name0 = ds_names[0]
            img_paths = batch["image_path"]

            id2mod = {i_: name for name, i_ in loader.dataset.modality_map.items()}

            class_mask = batch.get("class_mask", None)
            if class_mask is not None:
                class_mask = class_mask.to(device, non_blocking=True).float()

            for dsn in ds_names:
                dataset_counters[dsn] += 1

            dev = move_multitask_batch_to_device(batch, device)
            imgs = dev["imgs"]
            mod_ids = dev["mod_ids"]
            task_mask = dev["task_mask"]
            has_seg = dev["has_seg"]
            has_cls = dev["has_cls"]
            has_rep = dev["has_rep"]
            has_det = dev["has_det"]
            det_targets = dev["det_targets"]
            gt_boxes_for_roi = dev["gt_boxes_for_roi"]
            is_lss_mri = dev["is_lss_mri"]

            B = imgs.shape[0]

            outputs = model(
                imgs,
                modality_ids=mod_ids,
                task_mask=task_mask,
                boxes=gt_boxes_for_roi,
                det_targets=det_targets,
                captions=None,
                use_pred_boxes_for_mri_cls=False,
                det_score_thresh=0.3,
                is_lss_mri=is_lss_mri,
                # GT descriptors consumed only in specialist mode (model.use_gt_descriptors=True)
                measurements=(batch["measurements"].to(device, non_blocking=True) if "measurements" in batch else None),
                measurements_present=(batch["measurements_present"].to(device, non_blocking=True) if "measurements_present" in batch else None),
                measurement_sizes=batch.get("measurement_sizes"),
                busbra_seg_channel_indices=batch.get("busbra_seg_channel_indices"),
                has_birads=(batch["has_birads"].to(device, non_blocking=True) if "has_birads" in batch else None),
                has_pathology=(batch["has_pathology"].to(device, non_blocking=True) if "has_pathology" in batch else None),
                seg_mask_gt=(batch["seg_mask"].to(device, non_blocking=True) if "seg_mask" in batch else None),
                gt_birads_idx=(batch["birads_index"].to(device, non_blocking=True) if "birads_index" in batch else None),
                gt_pathology_idx=(batch["pathology_index"].to(device, non_blocking=True) if "pathology_index" in batch else None),
                compute_gradif_reference=True,
            )

            pred_boxes_list, pred_labels_list, pred_scores_list, pred_query_ids_list = get_det_predictions_for_eval(
                model=model,
                outputs=outputs,
                imgs=imgs,
                score_thresh=0.3,
                topk=20,
                nms_iou=0.4,
            )

            # ---------------------------------------------------------
            # 1a. BI-RADS + PATHOLOGY (BUS-BRA only) (added)
            # ---------------------------------------------------------
            if outputs.get('birads_logits') is not None and 'has_birads' in batch:
                _hb = batch['has_birads']
                if _hb.any():
                    _bgt = batch['birads_index']
                    _blogits = outputs['birads_logits'].float()
                    _thr = _model_birads_thresholds(model, _blogits.device)
                    _bpred = birads_decode(_blogits, _thr).cpu()
                    _bprob = birads_class_probabilities(_blogits).cpu()
                    _base_blogits = outputs.get('birads_base_logits', None)
                    if _base_blogits is not None:
                        _base_bpred = birads_decode(_base_blogits.float(), _thr).cpu()
                        _base_bprob = birads_class_probabilities(
                            _base_blogits.float()
                        ).cpu()
                    else:
                        _base_bpred, _base_bprob = _bpred, _bprob
                    for _bi2 in _hb.nonzero(as_tuple=True)[0].tolist():
                        _g = int(_bgt[_bi2])
                        if _g < 0:
                            continue
                        birads_pred_test.append(int(_bpred[_bi2]))
                        birads_gt_test.append(_g)
                        birads_prob_test.append(_bprob[_bi2].numpy())
                        birads_base_pred_test.append(int(_base_bpred[_bi2]))
                        birads_base_prob_test.append(_base_bprob[_bi2].numpy())

            if outputs.get('pathology_meas_logits') is not None and 'has_pathology' in batch and 'pathology_index' in batch:
                _hp = batch['has_pathology']
                _pidx_gt = batch['pathology_index']
                _valid_p = _hp & (_pidx_gt >= 0)
                if _valid_p.any():
                    _path_probs = torch.softmax(outputs['pathology_meas_logits'].float(), dim=1).cpu()
                    _path_pred = _path_probs.argmax(dim=1)
                    _base_path_logits = outputs.get('pathology_base_logits', None)
                    if _base_path_logits is not None:
                        _base_path_probs = torch.softmax(
                            _base_path_logits.float(), dim=1
                        ).cpu()
                        _base_path_pred = _base_path_probs.argmax(dim=1)
                    else:
                        _base_path_probs, _base_path_pred = _path_probs, _path_pred
                    for _pi2 in _valid_p.nonzero(as_tuple=True)[0].tolist():
                        pathology_pred_test.append(int(_path_pred[_pi2]))
                        pathology_gt_test.append(int(_pidx_gt[_pi2]))
                        pathology_prob_test.append(float(_path_probs[_pi2, 1]))
                        pathology_base_pred_test.append(int(_base_path_pred[_pi2]))
                        pathology_base_prob_test.append(float(_base_path_probs[_pi2, 1]))

            # ---------------------------------------------------------
            # 1. SEGMENTATION
            # ---------------------------------------------------------
            if any(has_seg) and outputs.get("pred_masks", None) is not None:
                H, W = imgs.shape[-2:]
                preds_interp = nn.functional.interpolate(outputs["pred_masks"], size=(H, W), mode="bilinear")
                pred_bin = (torch.sigmoid(preds_interp) > 0.5).float()
                _base_full = outputs.get("pred_masks_base", None)
                _base_bin = None
                if _base_full is not None:
                    _base_interp = nn.functional.interpolate(
                        _base_full, size=(H, W), mode="bilinear"
                    )
                    _base_bin = (torch.sigmoid(_base_interp) > 0.5).float()
                gt = batch["seg_mask"].to(device, non_blocking=True).float()
                active_ch = batch["seg_active_ch"].to(device, non_blocking=True).float()

                for idx, valid in enumerate(has_seg):
                    if not valid:
                        continue

                    d_name = ds_names[idx]
                    p = pred_bin[idx]       # [C, H, W]
                    g = gt[idx]             # [C, H, W]
                    a = active_ch[idx]      # [C]

                    # Ensure stratified buckets exist for this dataset
                    results[d_name].setdefault("seg_dice", [])           # combined (kept for backward compat)
                    results[d_name].setdefault("seg_iou", [])
                    results[d_name].setdefault("seg_dice_with_gt", [])   # samples with non-empty GT
                    results[d_name].setdefault("seg_iou_with_gt", [])
                    results[d_name].setdefault("seg_dice_empty_gt", [])  # samples with empty GT (normals)
                    results[d_name].setdefault("seg_iou_empty_gt", [])
                    results[d_name].setdefault("seg_base_dice", [])
                    results[d_name].setdefault("seg_base_iou", [])

                    ch_dices = []
                    ch_ious = []
                    ch_dices_with_gt = []
                    ch_ious_with_gt = []
                    ch_dices_empty_gt = []
                    ch_ious_empty_gt = []

                    for c in range(p.shape[0]):
                        if a[c] < 0.5:
                            continue

                        pred_sum  = p[c].sum()
                        gt_sum    = g[c].sum()
                        inter_c   = (p[c] * g[c]).sum()
                        union_add = pred_sum + gt_sum
                        union_iou = union_add - inter_c

                        pred_empty = (pred_sum.item() == 0)
                        gt_empty   = (gt_sum.item()   == 0)

                        # True negative: both empty -> perfect prediction
                        if gt_empty and pred_empty:
                            d_c = torch.tensor(1.0, device=p.device)
                            i_c = torch.tensor(1.0, device=p.device)
                        # False positive: GT empty but pred not empty
                        elif gt_empty and not pred_empty:
                            d_c = torch.tensor(0.0, device=p.device)
                            i_c = torch.tensor(0.0, device=p.device)
                        # False negative: GT not empty but pred empty
                        elif (not gt_empty) and pred_empty:
                            d_c = torch.tensor(0.0, device=p.device)
                            i_c = torch.tensor(0.0, device=p.device)
                        # Normal case: both non-empty
                        else:
                            d_c = (2.0 * inter_c) / (union_add + 1e-6)
                            i_c = inter_c / (union_iou + 1e-6)

                        d_c_val = d_c.item()
                        i_c_val = i_c.item()

                        ch_dices.append(d_c_val)
                        ch_ious.append(i_c_val)

                        # Stratify per-channel into "with GT" vs "empty GT"
                        if gt_empty:
                            ch_dices_empty_gt.append(d_c_val)
                            ch_ious_empty_gt.append(i_c_val)
                        else:
                            ch_dices_with_gt.append(d_c_val)
                            ch_ious_with_gt.append(i_c_val)

                    dice = float(np.mean(ch_dices)) if ch_dices else 0.0
                    iou  = float(np.mean(ch_ious))  if ch_ious  else 0.0

                    # Combined (kept for backward compatibility)
                    results[d_name]["seg_dice"].append(dice)
                    results[d_name]["seg_iou"].append(iou)

                    if _base_bin is not None:
                        _bp = _base_bin[idx]
                        _base_ch_d, _base_ch_i = [], []
                        for c in range(_bp.shape[0]):
                            if a[c] < 0.5:
                                continue
                            _ps = _bp[c].sum(); _gs = g[c].sum()
                            _inter = (_bp[c] * g[c]).sum()
                            if _gs.item() == 0 and _ps.item() == 0:
                                _dc = _ic = 1.0
                            elif _gs.item() == 0 or _ps.item() == 0:
                                _dc = _ic = 0.0
                            else:
                                _dc = float((2.0 * _inter / (_ps + _gs + 1e-6)).item())
                                _ic = float((_inter / (_ps + _gs - _inter + 1e-6)).item())
                            _base_ch_d.append(_dc); _base_ch_i.append(_ic)
                        if _base_ch_d:
                            results[d_name]["seg_base_dice"].append(float(np.mean(_base_ch_d)))
                            results[d_name]["seg_base_iou"].append(float(np.mean(_base_ch_i)))

                    # Stratified per-sample (sample-level mean of its own valid channels)
                    if len(ch_dices_with_gt) > 0:
                        results[d_name]["seg_dice_with_gt"].append(float(np.mean(ch_dices_with_gt)))
                        results[d_name]["seg_iou_with_gt"].append(float(np.mean(ch_ious_with_gt)))
                    if len(ch_dices_empty_gt) > 0:
                        results[d_name]["seg_dice_empty_gt"].append(float(np.mean(ch_dices_empty_gt)))
                        results[d_name]["seg_iou_empty_gt"].append(float(np.mean(ch_ious_empty_gt)))

                    # Optional: mask predictions to active channels before saving (cleaner visuals)
                    pred_for_vis = outputs["pred_masks"][idx].clone()
                    pred_for_vis[a < 0.5] = -1e4   # forces sigmoid to ~0 in inactive channels

                    if save_artifacts:
                        manager.save_seg_sample(
                            imgs[idx],
                            pred_for_vis,
                            gt[idx],
                            dice,
                            i * len(imgs) + idx,
                            d_name,
                        )

            # ---------------------------------------------------------
            # 2. CLASSIFICATION
            # ---------------------------------------------------------
            preds_bin_rep = None
            batch_gt_tags_list = [None] * B

            if outputs.get("class_logits", None) is not None and "class_labels" in batch:
                logits_all = outputs["class_logits"].float()

                for local_idx, is_valid in enumerate(has_cls):
                    if not is_valid:
                        continue

                    d_name = ds_names[local_idx]
                    tgt_full = batch["class_labels"][local_idx].float()

                    # ----------------------------
                    # A. MRI ROI CLS dataset = TRUE MULTICLASS
                    # ----------------------------
                    if d_name == "LSS_MRI_ROI_CLS":
                        roi_logits = logits_all[local_idx, severity_col_idx]      # [3]
                        roi_prob = torch.softmax(roi_logits, dim=0)               # [3]
                        tgt_roi = tgt_full[severity_col_idx]                      # [3]

                        if tgt_roi.sum().item() <= 0:
                            continue

                        tgt_id = int(torch.argmax(tgt_roi).item())
                        batch_gt_tags_list[local_idx] = severity_names_resolved[tgt_id]

                        cls_data[d_name]["probs"].append(roi_prob.detach().cpu())                     # [3]
                        cls_data[d_name]["targets"].append(torch.tensor(tgt_id, dtype=torch.long))   # scalar
                        cls_data[d_name]["masks"].append(torch.ones(len(severity_col_idx)))          # dummy

                        if save_artifacts:
                            manager.save_cls_sample(
                                imgs[local_idx],
                                tgt_roi.detach().cpu(),
                                roi_prob.detach().cpu(),
                                severity_names_resolved,
                                i * len(imgs) + local_idx,
                                d_name,
                            )

                    # ----------------------------
                    # B. Generic masked multilabel path
                    # ----------------------------
                    else:
                        prob = torch.sigmoid(logits_all[local_idx])

                        if class_mask is not None:
                            m = class_mask[local_idx]
                            prob_m = prob * m
                        else:
                            m = torch.ones_like(prob)
                            prob_m = prob

                        gt_indices = np.where(tgt_full.cpu().numpy() == 1)[0]
                        gt_tags = [class_names[c_idx] for c_idx in gt_indices]
                        batch_gt_tags_list[local_idx] = ", ".join(gt_tags)

                        cls_data[d_name]["probs"].append(prob_m.detach().cpu())
                        cls_data[d_name]["targets"].append(tgt_full.detach().cpu())
                        cls_data[d_name]["masks"].append(m.detach().cpu())

                        if save_artifacts:
                            manager.save_cls_sample(
                                imgs[local_idx],
                                tgt_full.detach().cpu(),
                                prob_m.detach().cpu(),
                                class_names,
                                i * len(imgs) + local_idx,
                                d_name,
                            )

                # only needed for NON-MRI report guidance
                probs = torch.sigmoid(logits_all)
                if class_mask is not None:
                    probs = probs * class_mask
                preds_bin_rep = (probs > 0.5).int().cpu().numpy()

            # ---------------------------------------------------------
            # 2.5 DETECTION + MRI ROI CROP CLASSIFIER
            # ---------------------------------------------------------
            mri_report_tags_by_sample = {}

            if det_targets is not None:
                # ----------------------------
                # A. detection metrics + save samples
                # ----------------------------
                for idx, valid in enumerate(has_det):
                    if not valid:
                        continue

                    d_name = ds_names[idx]
                    gt_boxes_b = det_targets[idx]["boxes"]
                    gt_labels_b = det_targets[idx]["labels"]

                    pred_boxes_abs = pred_boxes_list[idx]
                    pred_labels_b = pred_labels_list[idx]
                    pred_scores_b = pred_scores_list[idx]

                    det_data[d_name]["pred_boxes"].append(pred_boxes_abs.detach().cpu())
                    det_data[d_name]["pred_labels"].append(pred_labels_b.detach().cpu())
                    det_data[d_name]["pred_scores"].append(pred_scores_b.detach().cpu())
                    det_data[d_name]["gt_boxes"].append(gt_boxes_b.detach().cpu())
                    det_data[d_name]["gt_labels"].append(gt_labels_b.detach().cpu())

                    matches = match_detection_predictions_to_gt_class_aware(
                        pred_boxes_abs,
                        pred_labels_b,
                        gt_boxes_b,
                        gt_labels_b,
                        iou_thresh=0.5,
                    )

                    matched_pred = set()
                    matched_gt = set()

                    for p_idx, g_idx, iou in matches:
                        matched_pred.add(p_idx)
                        matched_gt.add(g_idx)
                        det_data[d_name]["tp"] += 1
                        det_data[d_name]["ious"].append(iou)

                    det_data[d_name]["fp"] += max(0, pred_boxes_abs.shape[0] - len(matched_pred))
                    det_data[d_name]["fn"] += max(0, gt_boxes_b.shape[0] - len(matched_gt))

                    if save_artifacts:
                        manager.save_det_sample(
                            img_tensor=imgs[idx],
                            pred_boxes=pred_boxes_abs.detach().cpu(),
                            pred_labels=pred_labels_b.detach().cpu(),
                            pred_scores=pred_scores_b.detach().cpu(),
                            gt_boxes=gt_boxes_b.detach().cpu(),
                            gt_labels=gt_labels_b.detach().cpu(),
                            idx=i * len(imgs) + idx,
                            ds_name=d_name,
                            det_class_names=det_class_names,
                        )

                # ----------------------------
                # B. detection -> crop classifier
                # ----------------------------
                crop_batch, crop_meta = extract_mri_roi_crops_from_batch(
                    imgs=imgs,
                    pred_boxes_list=pred_boxes_list,
                    out_size=loader.dataset.target_size if hasattr(loader.dataset, "target_size") else (224, 224),
                    expand_ratio=0.20,
                )

                if crop_batch is not None and len(crop_meta) > 0:
                    Ncrop = crop_batch.shape[0]

                    # run crop classifier through generic classification branch
                    mri_mod_id = loader.dataset.modality_map["MRI"]
                    crop_mod_ids = torch.full((Ncrop,), mri_mod_id, dtype=torch.long, device=device)
                    crop_task_mask = torch.zeros((Ncrop, 4), dtype=torch.float32, device=device)
                    crop_task_mask[:, 1] = 1.0
                    crop_is_lss_mri = torch.zeros((Ncrop,), dtype=torch.bool, device=device)

                    crop_outputs = model(
                        crop_batch.to(device, non_blocking=True),
                        modality_ids=crop_mod_ids,
                        task_mask=crop_task_mask,
                        boxes=None,
                        det_targets=None,
                        captions=None,
                        report_labels=None,
                        use_pred_boxes_for_mri_cls=False,
                        det_score_thresh=0.3,
                        is_lss_mri=crop_is_lss_mri,
                    )

                    crop_logits_all = crop_outputs["class_logits"].float()[:, severity_col_idx]  # [Ncrop, 3]
                    crop_probs_all = torch.softmax(crop_logits_all, dim=1)
                    crop_pred_ids_all = crop_probs_all.argmax(dim=1)

                    crop_pred_by_sample = defaultdict(list)
                    crop_prob_by_sample = defaultdict(list)
                    crop_logit_by_sample = defaultdict(list)

                    for n, (b_idx, pred_idx) in enumerate(crop_meta):
                        crop_pred_by_sample[b_idx].append((pred_idx, int(crop_pred_ids_all[n].item())))
                        crop_prob_by_sample[b_idx].append((pred_idx, crop_probs_all[n].detach().cpu()))
                        crop_logit_by_sample[b_idx].append((pred_idx, crop_logits_all[n].detach().cpu()))

                    for b in range(B):
                        if not has_det[b]:
                            continue

                        d_name = ds_names[b]
                        gt_boxes_b = det_targets[b]["boxes"]
                        gt_level_labels_b = det_targets[b]["labels"]
                        gt_sev_b = det_targets[b].get("severity", None)

                        pred_boxes_abs = pred_boxes_list[b]
                        pred_labels_b = pred_labels_list[b]

                        if gt_sev_b is None:
                            continue
                        if pred_boxes_abs is None or pred_boxes_abs.numel() == 0 or gt_boxes_b.numel() == 0:
                            continue

                        num_preds = pred_boxes_abs.shape[0]
                        pred_sev_ids = torch.full((num_preds,), -1, dtype=torch.long)
                        pred_sev_probs = torch.zeros((num_preds, len(severity_col_idx)), dtype=torch.float32)
                        pred_sev_logits = torch.zeros((num_preds, len(severity_col_idx)), dtype=torch.float32)

                        if b in crop_pred_by_sample:
                            for pred_idx, sev_id in crop_pred_by_sample[b]:
                                if 0 <= pred_idx < num_preds:
                                    pred_sev_ids[pred_idx] = sev_id
                            for pred_idx, prob_vec in crop_prob_by_sample[b]:
                                if 0 <= pred_idx < num_preds:
                                    pred_sev_probs[pred_idx] = prob_vec
                            for pred_idx, logit_vec in crop_logit_by_sample[b]:
                                if 0 <= pred_idx < num_preds:
                                    pred_sev_logits[pred_idx] = logit_vec

                        # tags for MRI report generation
                        valid_pred_mask = pred_sev_ids >= 0
                        if valid_pred_mask.any():
                            mri_tags = build_mri_predicted_tags_from_crop(
                                pred_level_labels_b=pred_labels_b[valid_pred_mask].detach().cpu().tolist(),
                                pred_severity_ids_b=pred_sev_ids[valid_pred_mask].detach().cpu().tolist(),
                                det_class_names=det_class_names,
                                severity_names=severity_names_resolved,
                                skip_normal=True,
                            )
                        else:
                            mri_tags = []

                        mri_report_tags_by_sample[b] = mri_tags

                        # severity metrics on matched detections
                        matches = match_roi_boxes_to_gt_class_aware(
                            pred_boxes_abs,
                            pred_labels_b,
                            gt_boxes_b,
                            gt_level_labels_b,
                            iou_thresh=0.5,
                        )

                        if len(matches) == 0:
                            continue

                        pred_ids = torch.tensor([m[0] for m in matches], dtype=torch.long)
                        gt_ids = torch.tensor([m[1] for m in matches], dtype=torch.long)

                        valid_match_mask = pred_sev_ids[pred_ids] >= 0
                        if not valid_match_mask.any():
                            continue

                        pred_ids = pred_ids[valid_match_mask]
                        gt_ids = gt_ids[valid_match_mask]

                        matched_gt_sev = gt_sev_b[gt_ids].long().cpu()
                        matched_pred_sev = pred_sev_ids[pred_ids].long().cpu()
                        matched_prob = pred_sev_probs[pred_ids].cpu()
                        matched_logits = pred_sev_logits[pred_ids].cpu()
                        matched_boxes = pred_boxes_abs[pred_ids].cpu()
                        matched_gt_boxes = gt_boxes_b[gt_ids].cpu()

                        mri_roi_crop_data[d_name]["targets"].append(matched_gt_sev)
                        mri_roi_crop_data[d_name]["preds"].append(matched_pred_sev)
                        mri_roi_crop_data[d_name]["probs"].append(matched_prob)

                        if save_artifacts:
                            manager.save_mri_roi_sample(
                                img_tensor=imgs[b],
                                pred_boxes=matched_boxes,
                                pred_logits=matched_logits,
                                gt_boxes=matched_gt_boxes,
                                gt_severity=matched_gt_sev,
                                idx=i * len(imgs) + b,
                                ds_name=d_name,
                                severity_names=severity_names_resolved,
                            )

            # ---------------------------------------------------------
            # 3. REPORT GENERATION
            # ---------------------------------------------------------
            if outputs.get("visual_tokens", None) is not None and has_rep.any():
                for idx, valid in enumerate(has_rep):
                    if not valid:
                        continue

                    mod_id_i = int(mod_ids[idx].item())
                    mod_name = id2mod.get(mod_id_i, f"Mod{mod_id_i}")
                    d_name = ds_names[idx]

                    use_prompt_guidance = bool(getattr(model, "use_cross_roi_reasoning", True))
                    # ALWAYS initialize
                    active_tags = []
                    if not use_prompt_guidance:
                        content_str = "Describe the radiological findings. Report:"

                    elif d_name == "LSS_MRI_AISSLab":
                        active_tags = mri_report_tags_by_sample.get(idx, [])
                        if active_tags:
                            content_str = f"Detected abnormalities: {', '.join(active_tags)}. Report:"
                        else:
                            content_str = "No abnormality detected. Report:"

                    elif d_name == "BUS_BRA":
                        # Specialist mode -> GT descriptors; unified -> option-B predicted.
                        if bool(getattr(model, "use_gt_descriptors", False)):
                            _mvec = batch["measurements"][idx] if "measurements" in batch else []
                            _bidx = int(batch["birads_index"][idx].item()) if "birads_index" in batch else -1
                            _pidx = int(batch["pathology_index"][idx].item()) if "pathology_index" in batch else -1
                            _desc = loader.dataset.build_gt_descriptor_dict(_mvec, _bidx, _pidx)
                        else:
                            _desc = {}
                            _mu = outputs.get(
                                'report_measurements_used',
                                outputs.get('measurements_used', None),
                            )
                            if _mu is not None:
                                _mv = _mu[idx].tolist()
                                _desc['max_diameter_px'] = _mv[0]
                                _desc['perpendicular_diameter_px'] = _mv[1]
                                _bw, _bh = _mv[2], _mv[3]
                                _desc['orientation'] = ('wider-than-tall orientation' if _bw > _bh
                                                        else 'taller-than-wide orientation' if _bh > _bw
                                                        else 'approximately equal width and height')
                            _pml = outputs.get('pathology_meas_logits', None)
                            if _pml is not None:
                                _path_names = getattr(loader.dataset, 'pathology_class_names', ['benign', 'malignant'])
                                _pi = int(_pml[idx].argmax().item())
                                _desc['pathology'] = _path_names[_pi] if _pi < len(_path_names) else str(_pi)
                            _bl = outputs.get('birads_logits', None)
                            if _bl is not None:
                                _bnames = getattr(loader.dataset, 'birads_class_names', ['2', '3', '4', '5'])
                                _bcat = int(birads_decode(_bl[idx:idx + 1])[0].item())
                                _desc['birads'] = _bnames[_bcat] if _bcat < len(_bnames) else str(_bcat)
                        content_str = loader.dataset._build_descriptor_content(
                            _desc, enforce_pixel_units=True
                        )
                        active_tags = []
                    else:
                        active_indices = np.where(preds_bin_rep[idx] == 1)[0] if preds_bin_rep is not None else []
                        active_tags = [class_names[c_idx] for c_idx in active_indices]
                        if NORMAL_LABEL in active_tags:
                            active_tags.remove(NORMAL_LABEL)

                        if active_tags:
                            content_str = f"Detected abnormalities: {', '.join(active_tags)}. Report:"
                        else:
                            content_str = "No abnormality detected. Report:"

                    messages = [
                        {"role": "system", "content": f"You are an expert {mod_name} radiologist. Describe the findings."},
                        {"role": "user", "content": content_str},
                    ]
                    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device, non_blocking=True)

                    vis_feat = outputs["visual_tokens"][idx:idx + 1]
                    gen_ids = model.report_decoder.generate(
                        visual_features=vis_feat,
                        prompt_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        do_sample=False,
                        num_beams=3,
                    )
                    pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

                    if "Report:" in pred_text:
                        pred_text = pred_text.split("Report:")[-1].strip()
                    elif "findings." in pred_text[:20].lower():
                        pred_text = pred_text.split("findings.")[-1].strip()

                    gt_text = batch["report_text"][idx]

                    text_data[d_name]["preds"].append(pred_text)
                    text_data[d_name]["gts"].append(gt_text)

                    pred_tokens = pred_text.split()
                    gt_tokens = gt_text.split()

                    m_score = meteor_score([gt_tokens], pred_tokens) if len(pred_tokens) > 0 and len(gt_tokens) > 0 else 0.0
                    r_score = rouge_evaluator.score(gt_text, pred_text)["rougeL"].fmeasure

                    results[d_name]["meteor"].append(m_score)
                    results[d_name]["rouge"].append(r_score)

                    ai_tags_str = ", ".join(active_tags) if active_tags else NORMAL_LABEL
                    gt_tags_str = batch_gt_tags_list[idx] if batch_gt_tags_list[idx] is not None else "N/A"

                    report_export_data[d_name].append({
                        "Image_Path": img_paths[idx],
                        "GT_Labels": gt_tags_str,
                        "AI_Predicted_Labels": ai_tags_str,
                        "GT_report": gt_text,
                        "AI_report": pred_text,
                        "Match": gt_tags_str == ai_tags_str,
                    })

                    manager.save_rep_record(
                        image_path=img_paths[idx],
                        gt_text=gt_text,
                        pred_text=pred_text,
                        idx=i * len(imgs) + idx,
                        ds_name=d_name,
                        scores={"bleu": b_score, "meteor": m_score, "rouge": r_score},
                        extra={
                            "GT_Labels": gt_tags_str,
                            "AI_Predicted_Labels": ai_tags_str,
                            "Match": (gt_tags_str == ai_tags_str),
                        },
                    )

        for h in handles:
            h.remove()

    # ---------------------------------------------------------------------
    # detailed report export
    # ---------------------------------------------------------------------
    if save_artifacts:
        report_save_dir = os.path.join(manager.run_dir, "detailed_reports")
        os.makedirs(report_save_dir, exist_ok=True)

        for ds_name, entries in report_export_data.items():
            report_df = pd.DataFrame(entries)
            column_order = [
                "Image_Path",
                "Match",
                "GT_Labels",
                "AI_Predicted_Labels",
                "GT_report",
                "AI_report",
            ]
            existing_columns = [
                col for col in column_order if col in report_df.columns
            ]
            report_df = report_df[existing_columns]
            file_path = os.path.join(
                report_save_dir, f"Audit_Reports_{ds_name}.xlsx"
            )
            report_df.to_excel(file_path, index=False)
            print(
                f"[Exported] {len(entries)} audit samples saved to {file_path}"
            )

    # ---------------------------------------------------------------------
    # 4. FINAL AGGREGATION & PRINTING
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"{'DATASET':<25} | {'TASK':<10} | {'METRICS':<40}")
    print("=" * 60)
    excel_row = {"Epoch": "TEST"}

    all_datasets = set(
        list(results.keys())
        + list(cls_data.keys())
        + list(det_data.keys())
        + list(mri_roi_crop_data.keys())
        + list(text_data.keys())
    )
    if birads_gt_test or pathology_gt_test:
        all_datasets.add("BUS_BRA")

    for ds in sorted(list(all_datasets)):
        print_lines = []

        # ----------------------------
        # A. Segmentation
        # ----------------------------
        if "seg_dice" in results[ds] and len(results[ds]["seg_dice"]) > 0:
            avg_dice = np.mean(results[ds]["seg_dice"])
            avg_iou = np.mean(results[ds].get("seg_iou", [0.0]))

            # Stratified metrics
            dices_with_gt   = results[ds].get("seg_dice_with_gt", [])
            ious_with_gt    = results[ds].get("seg_iou_with_gt", [])
            dices_empty_gt  = results[ds].get("seg_dice_empty_gt", [])
            ious_empty_gt   = results[ds].get("seg_iou_empty_gt", [])

            n_with    = len(dices_with_gt)
            n_empty   = len(dices_empty_gt)
            n_total   = n_with + n_empty

            avg_dice_with_gt   = float(np.mean(dices_with_gt))  if n_with  > 0 else float("nan")
            avg_iou_with_gt    = float(np.mean(ious_with_gt))   if n_with  > 0 else float("nan")
            avg_dice_empty_gt  = float(np.mean(dices_empty_gt)) if n_empty > 0 else float("nan")
            avg_iou_empty_gt   = float(np.mean(ious_empty_gt))  if n_empty > 0 else float("nan")

            # Console output
            print_lines.append(
                f"SEG (overall): Dice={avg_dice:.4f}, IoU={avg_iou:.4f}  "
                f"[n_with_GT={n_with}, n_empty_GT={n_empty}, n_total={n_total}]"
            )
            _base_d = results[ds].get("seg_base_dice", [])
            _base_i = results[ds].get("seg_base_iou", [])
            if len(_base_d) == len(results[ds]["seg_dice"]) and _base_d:
                _base_d_mean = float(np.mean(_base_d))
                _base_i_mean = float(np.mean(_base_i))
                print_lines.append(
                    f"SEG GraDiF gain vs identity: Dice={avg_dice - _base_d_mean:+.4f}, "
                    f"IoU={avg_iou - _base_i_mean:+.4f}"
                )
                excel_row[f"{ds}_Dice_Base"] = _base_d_mean
                excel_row[f"{ds}_IoU_Base"] = _base_i_mean
                excel_row[f"{ds}_Dice_GradifGain"] = avg_dice - _base_d_mean
            if n_with > 0:
                print_lines.append(
                    f"SEG (with GT — lesion present): "
                    f"Dice={avg_dice_with_gt:.4f}, IoU={avg_iou_with_gt:.4f}  (n={n_with})"
                )
            if n_empty > 0:
                print_lines.append(
                    f"SEG (empty GT — normal/specificity): "
                    f"Dice={avg_dice_empty_gt:.4f}, IoU={avg_iou_empty_gt:.4f}  (n={n_empty})"
                )

            # Excel logging
            excel_row[f"{ds}_Dice"]            = avg_dice
            excel_row[f"{ds}_IoU"]             = avg_iou
            excel_row[f"{ds}_Dice_with_GT"]    = avg_dice_with_gt
            excel_row[f"{ds}_IoU_with_GT"]     = avg_iou_with_gt
            excel_row[f"{ds}_Dice_empty_GT"]   = avg_dice_empty_gt
            excel_row[f"{ds}_IoU_empty_GT"]    = avg_iou_empty_gt
            excel_row[f"{ds}_N_with_GT"]       = n_with
            excel_row[f"{ds}_N_empty_GT"]      = n_empty

        # ----------------------------
        # A2. BI-RADS + Pathology (BUS-BRA only) (added)
        # ----------------------------
        if ds == "BUS_BRA" and birads_gt_test:
            _bnames = getattr(loader.dataset, 'birads_class_names', None) or ["2", "3", "4", "5"]
            _bmetrics, _bcm, _bperclass = compute_birads_metrics(
                birads_gt_test, birads_pred_test, birads_prob_test,
                class_names=_bnames, prefix="",
            )

            print_lines.append(
                "BI-RADS: "
                f"Acc={_bmetrics['Acc']:.4f}, "
                f"BalAcc={_bmetrics['BalancedAcc']:.4f}, "
                f"MacroF1={_bmetrics['MacroF1']:.4f}, "
                f"MacroAUC={_bmetrics.get('MacroAUC_OvR', float('nan')):.4f}, "
                f"MAE={_bmetrics['MAE']:.4f}, "
                f"Within1={_bmetrics['Within1']:.4f}, "
                f"QWK={_bmetrics['QWK']:.4f} (n={len(birads_gt_test)})"
            )

            # Main paper metrics in the test Excel/JSON row.
            for _name, _value in _bmetrics.items():
                excel_row[f"{ds}_BIRADS_{_name}"] = _value

            for _row in _bperclass:
                print_lines.append(
                    "BI-RADS %s: P=%.4f R=%.4f F1=%.4f (n=%d)"
                    % (_row['name'], _row['precision'], _row['recall'], _row['f1'], _row['support'])
                )

            print(f"\n[TEST BI-RADS confusion matrix — {ds}] [rows=GT, cols=Pred]")
            print(_bcm)
            if len(birads_base_pred_test) == len(birads_gt_test):
                _bbm, _, _ = compute_birads_metrics(
                    birads_gt_test,
                    birads_base_pred_test,
                    birads_base_prob_test,
                    class_names=_bnames,
                    prefix="",
                )
                print_lines.append(
                    "BI-RADS GraDiF gain vs identity: "
                    f"MacroF1={_bmetrics['MacroF1'] - _bbm['MacroF1']:+.4f}, "
                    f"AUC={_bmetrics.get('MacroAUC_OvR', 0.0) - _bbm.get('MacroAUC_OvR', 0.0):+.4f}, "
                    f"QWK={_bmetrics['QWK'] - _bbm['QWK']:+.4f}"
                )

        if ds == "BUS_BRA" and pathology_gt_test:
            _pp = np.array(pathology_pred_test)
            _pg = np.array(pathology_gt_test)
            _pprob = np.array(pathology_prob_test)
            _path_names = getattr(loader.dataset, 'pathology_class_names', None) or ['benign', 'malignant']

            path_acc = float((_pp == _pg).mean())
            path_macro_f1 = float(f1_score(_pg, _pp, average='macro', zero_division=0))
            path_weighted_f1 = float(f1_score(_pg, _pp, average='weighted', zero_division=0))

            line = f"PATHOLOGY: Acc={path_acc:.4f}, MacroF1={path_macro_f1:.4f}, WeightedF1={path_weighted_f1:.4f} (n={len(_pg)})"
            excel_row[f"{ds}_Pathology_Acc"] = path_acc
            excel_row[f"{ds}_Pathology_MacroF1"] = path_macro_f1
            excel_row[f"{ds}_Pathology_WeightedF1"] = path_weighted_f1

            if set(_pg.tolist()) <= {0, 1}:
                _pos = (_pg == 1)
                _neg = (_pg == 0)
                if _pos.any():
                    sens = float((_pp[_pos] == 1).mean())
                    excel_row[f"{ds}_Pathology_Sensitivity"] = sens
                    line += f", Sens={sens:.4f}"
                if _neg.any():
                    spec = float((_pp[_neg] == 0).mean())
                    excel_row[f"{ds}_Pathology_Specificity"] = spec
                    line += f", Spec={spec:.4f}"
                if _pos.any() and _neg.any():
                    try:
                        path_auc = float(roc_auc_score(_pg, _pprob))
                        excel_row[f"{ds}_Pathology_AUC"] = path_auc
                        line += f", AUC={path_auc:.4f}"
                    except Exception as e:
                        print(f"  Could not compute Pathology AUC for {ds}: {e}")

            if len(pathology_base_pred_test) == len(pathology_gt_test):
                _pbp = np.array(pathology_base_pred_test)
                _pbprob = np.array(pathology_base_prob_test)
                _base_f1 = float(f1_score(_pg, _pbp, average='macro', zero_division=0))
                try:
                    _base_auc = float(roc_auc_score(_pg, _pbprob))
                except Exception:
                    _base_auc = 0.0
                _fused_auc = float(excel_row.get(f"{ds}_Pathology_AUC", 0.0))
                line += (
                    f" | GraDiF gain vs identity: MacroF1={path_macro_f1 - _base_f1:+.4f}, "
                    f"AUC={_fused_auc - _base_auc:+.4f}"
                )
            print_lines.append(line)

            for _c in sorted(set(_pg.tolist())):
                _m = (_pg == _c)
                _cat = _path_names[_c] if _c < len(_path_names) else str(_c)
                _acc_c = float((_pp[_m] == _pg[_m]).mean())
                excel_row[f"{ds}_Pathology_cat{_cat}_Acc"] = _acc_c

            try:
                pcm = confusion_matrix(_pg, _pp)
                print(f"\n[TEST Pathology confusion matrix — {ds}]")
                print(pcm)
            except Exception as e:
                print(f"  Could not compute pathology confusion matrix for {ds}: {e}")

        # ----------------------------
        # B1. LSS_MRI_ROI_CLS = MULTICLASS
        # ----------------------------
        if ds == "LSS_MRI_ROI_CLS" and ds in cls_data and len(cls_data[ds]["targets"]) > 0:
            y_prob = torch.stack(cls_data[ds]["probs"]).float().numpy()     # [N, 3]
            y_true = torch.stack(cls_data[ds]["targets"]).long().numpy()    # [N]
            y_pred = y_prob.argmax(axis=1)

            acc = accuracy_score(y_true, y_pred)
            macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
            macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

            roi_map, _ = compute_multiclass_map(y_true, y_prob, num_classes=y_prob.shape[1])

            try:
                macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            except Exception:
                macro_auc = 0.0

            cm = confusion_matrix(y_true, y_pred)

            print_lines.append(
                f"ROI-CLS: Acc={acc:.4f}, MacroAUC={macro_auc:.4f}, mAP={roi_map:.4f}, "
                f"P={macro_precision:.4f}, R={macro_recall:.4f}, "
                f"MacroF1={macro_f1:.4f}, WeightedF1={weighted_f1:.4f}"
            )

            excel_row[f"{ds}_Acc"] = acc
            excel_row[f"{ds}_MacroAUC"] = macro_auc
            excel_row[f"{ds}_mAP"] = roi_map
            excel_row[f"{ds}_PrecisionMacro"] = macro_precision
            excel_row[f"{ds}_RecallMacro"] = macro_recall
            excel_row[f"{ds}_MacroF1"] = macro_f1
            excel_row[f"{ds}_WeightedF1"] = weighted_f1

            print_lines.append(f"ROI-CLS CM:\n{cm}")

            if save_artifacts:
                report_dir = os.path.join(
                    manager.run_dir, "classification_reports"
                )
                os.makedirs(report_dir, exist_ok=True)
                report_path = os.path.join(
                    report_dir, f"report_cls_{ds}.txt"
                )
                with open(report_path, "w") as f:
                    f.write(f"Classification Report for {ds}\n\n")
                    f.write(classification_report(
                        y_true,
                        y_pred,
                        target_names=severity_names_resolved,
                        digits=4,
                        zero_division=0
                    ))
                    f.write("\nConfusion Matrix:\n")
                    f.write(np.array2string(cm))
                print_lines.append(
                    "     (Multiclass report saved to "
                    f"{os.path.basename(report_path)})"
                )

        # ----------------------------
        # B2. Generic masked multilabel classification
        # ----------------------------
        elif ds in cls_data and len(cls_data[ds].get("targets", [])) > 0:
            y_probs = torch.stack(cls_data[ds]["probs"]).numpy()
            y_true = torch.stack(cls_data[ds]["targets"]).numpy()
            y_mask = torch.stack(cls_data[ds]["masks"]).numpy()
            y_pred = ((y_probs > 0.5) & (y_mask == 1)).astype(int)

            C = y_true.shape[1]
            auc_list, ap_list, f1_list = [], [], []

            for c in range(C):
                valid = y_mask[:, c] == 1
                if valid.sum() < 5:
                    continue

                y_c = y_true[valid, c]
                p_c = y_probs[valid, c]
                yhat_c = y_pred[valid, c]

                if len(np.unique(y_c)) < 2:
                    continue

                auc_list.append(roc_auc_score(y_c, p_c))
                ap_list.append(average_precision_score(y_c, p_c))
                f1_list.append(f1_score(y_c, yhat_c))

            macro_auc = float(np.mean(auc_list)) if auc_list else 0.0
            mAP = float(np.mean(ap_list)) if ap_list else 0.0
            macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

            valid_all = y_mask == 1
            y_true_flat = y_true[valid_all]
            y_pred_flat = y_pred[valid_all]
            micro_f1 = f1_score(y_true_flat, y_pred_flat, average="micro") if y_true_flat.size else 0.0
            hamming = hamming_loss(y_true_flat, y_pred_flat) if y_true_flat.size else 0.0

            print_lines.append(
                f"CLS: M-AUC={macro_auc:.4f}, mAP={mAP:.4f}, "
                f"M-F1={macro_f1:.4f}, microF1={micro_f1:.4f}, Hamm={hamming:.4f}"
            )

            excel_row[f"{ds}_MacroAUC"] = macro_auc
            excel_row[f"{ds}_mAP"] = mAP
            excel_row[f"{ds}_MacroF1"] = macro_f1
            excel_row[f"{ds}_MicroF1"] = micro_f1
            excel_row[f"{ds}_Hamming"] = hamming

            if save_artifacts:
                ds_auc = manager.plot_dataset_roc_masked(
                    y_true, y_probs, y_mask, class_names, ds
                )
                manager.plot_dataset_cm_masked(
                    y_true, y_pred, y_mask, class_names, ds
                )
            else:
                ds_auc = macro_auc

            valid = (y_mask == 1)
            y_pred_bin = ((y_probs > 0.5) & valid).astype(int)
            y_true_bin = y_true.astype(int)
            y_pred_flat = y_pred_bin[valid]
            y_true_flat = y_true_bin[valid]
            acc_flat = (y_pred_flat == y_true_flat).mean() if y_true_flat.size else 0.0

            print_lines.append(f"CLS: Acc(Flat,Masked)={acc_flat:.4f}, AUC={ds_auc:.4f}")
            excel_row[f"{ds}_AccFlatMasked"] = acc_flat
            excel_row[f"{ds}_AUC"] = ds_auc

        # ----------------------------
        # DETECTION
        # ----------------------------
        if ds in det_data and len(det_data[ds]["gt_boxes"]) > 0:
            tp = det_data[ds]["tp"]
            fp = det_data[ds]["fp"]
            fn = det_data[ds]["fn"]

            det_precision = tp / (tp + fp + 1e-8)
            det_recall = tp / (tp + fn + 1e-8)
            det_f1 = 2.0 * det_precision * det_recall / (det_precision + det_recall + 1e-8)
            det_mean_iou = float(np.mean(det_data[ds]["ious"])) if len(det_data[ds]["ious"]) > 0 else 0.0
            det_map50 = evaluate_detection_map50_from_storage(det_data[ds], det_class_names)

            print_lines.append(
                f"DET: mAP@50={det_map50:.4f}, P={det_precision:.4f}, "
                f"R={det_recall:.4f}, F1={det_f1:.4f}, IoU={det_mean_iou:.4f}"
            )

            excel_row[f"{ds}_Det_mAP50"] = det_map50
            excel_row[f"{ds}_Det_Precision"] = det_precision
            excel_row[f"{ds}_Det_Recall"] = det_recall
            excel_row[f"{ds}_Det_F1"] = det_f1
            excel_row[f"{ds}_Det_MeanIoU"] = det_mean_iou

        # ----------------------------
        # MRI detection -> crop classifier
        # ----------------------------
        if ds in mri_roi_crop_data and len(mri_roi_crop_data[ds]["targets"]) > 0:
            roi_gt = torch.cat(mri_roi_crop_data[ds]["targets"], dim=0).long().numpy()
            roi_pred = torch.cat(mri_roi_crop_data[ds]["preds"], dim=0).long().numpy()
            roi_prob = torch.cat(mri_roi_crop_data[ds]["probs"], dim=0).float().numpy()

            roi_acc = float((roi_pred == roi_gt).mean()) if len(roi_gt) > 0 else 0.0
            roi_macro_f1 = f1_score(roi_gt, roi_pred, average="macro", zero_division=0) if len(roi_gt) > 0 else 0.0
            roi_weighted_f1 = f1_score(roi_gt, roi_pred, average="weighted", zero_division=0) if len(roi_gt) > 0 else 0.0
            roi_macro_precision = precision_score(roi_gt, roi_pred, average="macro", zero_division=0) if len(roi_gt) > 0 else 0.0
            roi_macro_recall = recall_score(roi_gt, roi_pred, average="macro", zero_division=0) if len(roi_gt) > 0 else 0.0
            roi_map, _ = compute_multiclass_map(roi_gt, roi_prob, num_classes=roi_prob.shape[1])

            print_lines.append(
                f"MRI-ROI-CROP: Acc={roi_acc:.4f}, mAP={roi_map:.4f}, "
                f"P={roi_macro_precision:.4f}, R={roi_macro_recall:.4f}, "
                f"MacroF1={roi_macro_f1:.4f}, WeightedF1={roi_weighted_f1:.4f}"
            )

            excel_row[f"{ds}_MRI_ROI_CROP_Acc"] = roi_acc
            excel_row[f"{ds}_MRI_ROI_CROP_mAP"] = roi_map
            excel_row[f"{ds}_MRI_ROI_CROP_Precision"] = roi_macro_precision
            excel_row[f"{ds}_MRI_ROI_CROP_Recall"] = roi_macro_recall
            excel_row[f"{ds}_MRI_ROI_CROP_MacroF1"] = roi_macro_f1
            excel_row[f"{ds}_MRI_ROI_CROP_WeightedF1"] = roi_weighted_f1

        # ----------------------------
        # REPORT GENERATION
        # ----------------------------
        if ds in text_data and len(text_data[ds]["preds"]) > 0:
            preds = text_data[ds]["preds"]
            gts = text_data[ds]["gts"]

            meteor = np.mean(results[ds]["meteor"]) if len(results[ds]["meteor"]) > 0 else 0.0
            rouge = np.mean(results[ds]["rouge"]) if len(results[ds]["rouge"]) > 0 else 0.0

            print_lines.append(
                f"REP: METEOR={meteor:.3f}, ROUGE-L={rouge:.3f}"
            )

            excel_row[f"{ds}_METEOR"] = meteor
            excel_row[f"{ds}_ROUGE_L"] = rouge

        if not print_lines:
            continue

        print(f"{ds:<25} | {print_lines[0]}")
        for line in print_lines[1:]:
            print(f"{'':<25} | {line}")
        print("-" * 60)

    # ---------------------------------------------------------------------
    # Save to Excel
    # ---------------------------------------------------------------------
    if append_to_log:
        df = pd.read_excel(manager.log_file)
        df = pd.concat([df, pd.DataFrame([excel_row])], ignore_index=True)
        df.to_excel(manager.log_file, index=False)
        print("\n[Saved] Detailed per-dataset metrics saved to excel.")

    return dict(excel_row)


class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self._clahe = None 

    def __call__(self, image):  
        if getattr(self, '_clahe', None) is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, 
                tileGridSize=self.tile_grid_size
            )
        
        img_np = np.array(image)
        
        if len(img_np.shape) == 3:  # RGB
            # Convert to LAB (Luminance, A, B)
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            # Apply CLAHE only to the L channel (brightness) to preserve color
            img_lab[:, :, 0] = self._clahe.apply(img_lab[:, :, 0])
            img_np = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
        else:  # Grayscale
            img_np = self._clahe.apply(img_np)
            
        return Image.fromarray(img_np)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits, target, active_channels=None):
        """
        pred_logits: [B, C, H, W]
        target:      [B, C, H, W]
        active_channels: [B, C] — 1.0 for supervised channels, 0.0 for don't-care
        """
        probs = torch.sigmoid(pred_logits)

        # Per-channel dice
        probs_flat  = probs.reshape(probs.shape[0], probs.shape[1], -1)    # [B, C, H*W]
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)  # [B, C, H*W]

        intersection = (probs_flat * target_flat).sum(dim=2)                # [B, C]
        union = probs_flat.sum(dim=2) + target_flat.sum(dim=2)              # [B, C]
        dice_per_ch = (2 * intersection + self.smooth) / (union + self.smooth)  # [B, C]

        if active_channels is not None:
            # Only average over channels this sample actually supervises
            dice_per_ch = dice_per_ch * active_channels                     # zero out irrelevant
            num_active = active_channels.sum(dim=1).clamp(min=1)            # [B]
            dice_per_sample = dice_per_ch.sum(dim=1) / num_active           # [B]
        else:
            dice_per_sample = dice_per_ch.mean(dim=1)                       # [B]

        return 1 - dice_per_sample.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean': return torch.mean(F_loss)
        elif self.reduction == 'sum': return torch.sum(F_loss)
        else: return F_loss

def masked_bce_with_logits(pred, target, active_channels, suppression_weight=0.1):
    """
    BCE on channels this sample supervises, plus a light suppression term
    that pushes inactive channels toward zero (negative logits).

    Why suppression: without it, channels not relevant to a sample's dataset
    receive zero gradient and their logits drift to random values, producing
    noisy/speckled false predictions at inference time on those channels.

    pred:             [B, C, H, W]  raw logits
    target:           [B, C, H, W]  binary targets (only meaningful where active_channels==1)
    active_channels:  [B, C]        1.0 = supervise this channel, 0.0 = inactive (suppress)
    suppression_weight: float       weight of the inactive-channel suppression term
                                    (0.1 is a safe default; raise to 0.2-0.3 if speckle persists,
                                    lower to 0.05 if it hurts supervised-channel quality)
    """
    B, C, H, W = pred.shape
    HW = float(H * W)

    # Per-pixel BCE for the supervised positive task
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')  # [B, C, H, W]
    ch_mask = active_channels.unsqueeze(-1).unsqueeze(-1)                     # [B, C, 1, 1]

    # ---- 1. Active-channel loss (the original BCE on supervised channels) ----
    masked = bce * ch_mask
    active_loss = masked.sum() / (ch_mask.sum() * HW + 1e-8)

    # ---- 2. Inactive-channel suppression loss ----
    # For channels this sample does NOT supervise, push predictions toward 0.
    # This prevents the "unsupervised channels drift to random" problem
    # that causes scattered/peppered false positives at test time.
    inactive_mask = (1.0 - active_channels).unsqueeze(-1).unsqueeze(-1)       # [B, C, 1, 1]
    inactive_count = inactive_mask.sum()

    if inactive_count.item() > 0:
        zero_target = torch.zeros_like(pred)
        suppression = F.binary_cross_entropy_with_logits(
            pred, zero_target, reduction='none'
        )                                                                      # [B, C, H, W]
        suppression_loss = (suppression * inactive_mask).sum() / (inactive_count * HW + 1e-8)
    else:
        suppression_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

    return active_loss + suppression_weight * suppression_loss


def segmentation_loss_per_sample(pred, target, active_channels, suppression_weight=0.1):
    """Dice+BCE per sample for the spatial GraDiF no-harm objective."""
    B, C, H, W = pred.shape
    ch = active_channels.float()
    ch4 = ch.unsqueeze(-1).unsqueeze(-1)
    active_count = ch.sum(dim=1).clamp_min(1.0)

    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    active_bce = (bce * ch4).sum(dim=(1, 2, 3)) / (
        active_count * float(H * W)
    )

    inactive = 1.0 - ch
    inactive_count = inactive.sum(dim=1)
    suppress = F.binary_cross_entropy_with_logits(
        pred, torch.zeros_like(pred), reduction='none'
    )
    inactive_loss = (
        suppress * inactive.unsqueeze(-1).unsqueeze(-1)
    ).sum(dim=(1, 2, 3)) / (
        inactive_count.clamp_min(1.0) * float(H * W)
    )
    inactive_loss = torch.where(
        inactive_count > 0, inactive_loss, torch.zeros_like(inactive_loss)
    )

    probs = torch.sigmoid(pred)
    probs_f = probs.reshape(B, C, -1)
    target_f = target.reshape(B, C, -1)
    inter = (probs_f * target_f).sum(dim=-1)
    denom = probs_f.sum(dim=-1) + target_f.sum(dim=-1)
    dice_ch = (2.0 * inter + 1e-6) / (denom + 1e-6)
    dice_sample = (dice_ch * ch).sum(dim=1) / active_count
    return (1.0 - dice_sample) + active_bce + suppression_weight * inactive_loss


def save_expert_heatmap(model, loader, device, save_path, class_names_map, max_samples_per_mod=30):
    """
    Robust heatmap generator that hunts for ALL modalities and saves to both PNG and Excel.
    """
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    
    # Create the Excel filename based on the save_path
    # e.g., "logs/heatmap.png" -> "logs/heatmap.xlsx"
    excel_path = os.path.splitext(save_path)[0] + ".xlsx"

    model.eval()
    records = []
    tasks = ['Segmentation', 'Classification', 'Reporting', 'Detection']
    
    # Track which modalities we have found
    found_counts = {k: 0 for k in class_names_map.values()}
    
    # We will stop when we have enough samples for ALL modalities
    total_scanned = 0
    
    with torch.no_grad():
        for batch in loader:
            task_mask = batch['task_mask'].to(device, non_blocking=True)
            total_scanned += 1
            if total_scanned > 50: break # Safety break
            
            # Check if we need anything from this batch
            mod_ids = batch['modality_id']
            batch_needs_processing = False
            
            for m_id in mod_ids:
                m_name = class_names_map.get(m_id.item(), 'Unknown')
                if found_counts.get(m_name, 0) < max_samples_per_mod:
                    batch_needs_processing = True
                    break
            
            if not batch_needs_processing: continue

            # --- Process Batch ---
            imgs = batch['image'].to(device, non_blocking=True)
            mod_ids = mod_ids.to(device, non_blocking=True)
            B = imgs.shape[0]
            outputs = model(imgs, modality_ids=mod_ids, task_mask=task_mask)
            
            if 'moe_probs' not in outputs: continue

            moe_probs_list = outputs['moe_probs']
            mod_ids_np = mod_ids.detach().cpu().numpy()
            task_mask_np = task_mask.detach().cpu().numpy()
            probs_np_list = [p.detach().cpu().numpy() for p in moe_probs_list]
            
            # Iterate through batch items
            for b in range(len(mod_ids_np)):
                mod_name = class_names_map.get(mod_ids_np[b], 'Unknown')

                if found_counts.get(mod_name, 0) >= max_samples_per_mod:
                    continue

                found_counts[mod_name] = found_counts.get(mod_name, 0) + 1

                for task_idx, probs_np in enumerate(probs_np_list):
                    # Skip inactive task/sample pairs
                    if task_mask_np[b, task_idx] <= 0:
                        continue

                    task_name = tasks[task_idx]

                    for expert_idx, prob_val in enumerate(probs_np[b]):
                        records.append({
                            'Condition': f"{mod_name} - {task_name}",
                            'Expert': f"E{expert_idx}",
                            'Probability': float(prob_val)
                        })
            
            # Check termination
            if all(c >= max_samples_per_mod for c in found_counts.values()):
                break

    # 2. Aggregation & Saving
    if not records: 
        print("[Vis] Warning: No records found for heatmap.")
        return

    df = pd.DataFrame(records)
    print(f"[Vis] Samples found: {found_counts}")

    # Create the Pivot Table (Rows=Condition, Cols=Expert, Values=Mean Prob)
    heatmap_data = df.pivot_table(index='Condition', columns='Expert', values='Probability', aggfunc='mean')

    num_experts = heatmap_data.shape[1] 
    expert_order = [f"E{i}" for i in range(num_experts)] # OR range(12) if fixed
    # Only keep columns that actually exist in data to avoid errors, or reindex carefully
    existing_cols = [c for c in expert_order if c in heatmap_data.columns]
    heatmap_data = heatmap_data[existing_cols]

    # Ensure ALL modalities appear (Stratified Rows)
    expected_rows = []
    for m in sorted(class_names_map.values()): # Sorted for clean look
        for t in tasks:
            expected_rows.append(f"{m} - {t}")
    
    # Reindex fills missing rows with NaN
    heatmap_data = heatmap_data.reindex(expected_rows)
    
    # Fill NaN with 0.0 for the Excel file (cleaner reading)
    heatmap_data_filled = heatmap_data.fillna(0.0)

    # --- SAVE EXCEL ---
    try:
        heatmap_data_filled.to_excel(excel_path, float_format="%.4f")
        print(f"[Vis] Saved Heatmap Data to Excel: {excel_path}")
    except Exception as e:
        print(f"[Vis] Error saving Excel: {e}")

    # --- SAVE IMAGE ---
    plt.figure(figsize=(14, 14)) # Taller figure
    sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f", vmin=0, vmax=1)
    plt.title("Router Specialization Snapshot (Stratified)")
    plt.tight_layout()
    plt.switch_backend('agg') 
    plt.savefig(save_path)
    plt.close()
    print(f"[Vis] Saved Heatmap Image: {save_path}")

def _norm_name(s: str) -> str:
    # Normalize labels between "Pleural Other" and "Pleural_Other"
    return str(s).strip().replace(" ", "_")

def compute_pos_weight_v3(
    train_ds,
    min_weight=0.3,
    max_weight=5.0,
    smoothing="log",          # "log" or "sqrt"
    smooth_scale=2.0,         # log scale multiplier (tune 1.5~3.0)
    problem_weight=0.6,       # multiply weight for problem classes (lower -> fewer FP)
    force_problem_to_min=False
):
    """
    FP-aware pos_weight:
    - coverage-aware denominator (potential_total)
    - smooths extreme neg/pos ratios
    - reduces weights for FP-heavy classes (and optionally forces to min)
    """
    num_classes = len(train_ds.class_names)
    pos = torch.zeros(num_classes, dtype=torch.float32)
    potential_total = torch.zeros(num_classes, dtype=torch.float32)

    # 1) Dataset -> which classes exist in that dataset (coverage map)
    ds_class_coverage = defaultdict(set)
    for ds_name, cfg in train_ds.config['datasets'].items():
        ds_classes = cfg.get('classes', cfg.get('class_names', []))
        ds_classes = [c.replace("No_Finding", train_ds.NORMAL_LABEL) for c in ds_classes]
        ds_class_coverage[ds_name] = set(_norm_name(c) for c in ds_classes)

    # 2) Count positives & potentials only where the dataset actually supports that class
    for item in train_ds.data:
        if 'classification' not in item.get('tasks', []):
            continue

        ds_name = item['ds_name']
        active_classes = ds_class_coverage.get(ds_name, set())

        labels = item['record']['annotations'].get('classification_labels', {})

        # Normalize keys once
        labels_norm = {}
        for k, v in labels.items():
            k2 = k.replace("No_Finding", train_ds.NORMAL_LABEL)
            labels_norm[_norm_name(k2)] = int(v)

        for class_name, class_idx in train_ds.class_map.items():
            cn = _norm_name(class_name)
            if cn in active_classes:
                potential_total[class_idx] += 1.0
                if labels_norm.get(cn, 0) == 1:
                    pos[class_idx] += 1.0

    # 3) Build raw pos_weight = neg/pos
    eps = 1e-6
    neg = torch.clamp(potential_total - pos, min=0.0)
    raw = (neg + 1.0) / (pos + 1.0)   # +1 Laplace to avoid blowups on rare classes

    # 4) Smooth extremes
    if smoothing == "sqrt":
        smoothed = torch.sqrt(raw)          # compress 100 -> 10
    else:
        smoothed = torch.log1p(raw) * smooth_scale  # compress more gently

    final_weight = torch.clamp(smoothed, min=min_weight, max=max_weight)

    # 5) FP-heavy problem classes: reduce their weights.
    # Only list classes that genuinely over-fire (high FP / high Specificity).
    # Do NOT include Consolidation, Fracture, Edema, Lung_Lesion, Lung_Opacity --
    # those have 0 sensitivity and need MORE weight, not less.
    problem_classes = [
        "Enlarged_Cardiomediastinum",
        "Pleural_Other",
        "No_Finding",
    ]
    problem_classes = set(_norm_name(c) for c in problem_classes)

    for cls, idx in train_ds.class_map.items():
        if _norm_name(cls) in problem_classes:
            if force_problem_to_min:
                final_weight[idx] = min_weight
            else:
                final_weight[idx] = torch.clamp(final_weight[idx] * problem_weight, min=min_weight, max=max_weight)

    # 6) If a class never appears in any dataset -> neutral weight
    final_weight[potential_total == 0] = 1.0
    return final_weight

class EqualModalityBatchSampler(Sampler):
    """Eight-slot sampler with equal contribution from all four modalities.

    Each batch contains:
      1-2. Two X-ray samples
      3-4. Two CT samples
      5-6. Two MRI samples
      7. One BUS-BRA sample, BI-RADS-balanced
      8. One BUS-BRA sample, pathology-balanced

    MRI pairs rotate across full-image, segmentation, and class-balanced ROI
    classification pools. The slot name controls only sampling: every selected
    record keeps all supervision signals stored in it. Repeated images within a
    modality are avoided whenever the corresponding pool has enough examples.
    """

    def __init__(
        self,
        dataset,
        batch_size=8,
        modality_batch_counts=None,
        batches_per_epoch=500,
        shuffle=True,
        drop_last=True,
        random_seed=42,
    ):
        expected_modality_counts = {
            "xray": 2,
            "ct": 2,
            "mri": 2,
            "ultrasound": 2,
        }
        actual_modality_counts = {
            str(name).lower(): int(value)
            for name, value in (
                modality_batch_counts or expected_modality_counts
            ).items()
        }
        if batch_size != 8 or actual_modality_counts != expected_modality_counts:
            raise ValueError(
                "Equal modality sampling requires batch_size=8 "
                "and modality_batch_counts={xray: 2, ct: 2, mri: 2, "
                "ultrasound: 2}."
            )

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.modality_batch_counts = actual_modality_counts
        self.batches_per_epoch = int(batches_per_epoch)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.base_seed = int(random_seed)
        self.epoch = 0
        self.rng = random.Random(self.base_seed)
        self._duplicate_warning_pools = set()

        if not hasattr(dataset, "data"):
            raise AttributeError("dataset.data not found")
        self.records = dataset.data

        self.mri_full_indices = []
        self.mri_seg_indices = []
        self.mri_roi_cls_indices = []
        self.xray_indices = []
        self.ct_indices = []
        self.report_indices_by_modality = {
            "xray": [],
            "ct": [],
            "mri": [],
            "ultrasound": [],
        }

        self.busbra_all_indices = []
        self.busbra_birads_by_class = {0: [], 1: [], 2: [], 3: []}
        self.busbra_pathology_by_class = {0: [], 1: []}

        # Route all records.  BUS-BRA is handled before generic modality
        # routing so its three protected slots cannot be diluted by other US.
        for idx, rec in enumerate(self.records):
            ds_name = str(rec.get("ds_name", "")).strip()
            mod_name = str(rec.get("mod_name", "")).strip().lower()
            tasks = {str(t).strip().lower() for t in rec.get("tasks", [])}

            if ds_name == "BUS_BRA":
                self.busbra_all_indices.append(idx)
                if self._record_has_report(idx):
                    self.report_indices_by_modality["ultrasound"].append(idx)
                ann = rec.get("record", {}).get("annotations", {}) or {}

                bir = ann.get("birads_classification", {}) or {}
                path = ann.get("pathology_classification", {}) or {}
                try:
                    birads_id = int(bir.get("class_index", -100))
                except (TypeError, ValueError):
                    birads_id = -100
                try:
                    pathology_id = int(path.get("class_index", -100))
                except (TypeError, ValueError):
                    pathology_id = -100

                if birads_id in self.busbra_birads_by_class:
                    self.busbra_birads_by_class[birads_id].append(idx)
                if pathology_id in self.busbra_pathology_by_class:
                    self.busbra_pathology_by_class[pathology_id].append(idx)

                continue

            if ds_name == "LSS_MRI_ROI_CLS":
                self.mri_roi_cls_indices.append(idx)
                if self._record_has_report(idx):
                    self.report_indices_by_modality["mri"].append(idx)
                continue

            if ds_name == "LSS_MRI_AISSLab":
                if "segmentation" in tasks:
                    self.mri_seg_indices.append(idx)
                else:
                    self.mri_full_indices.append(idx)
                if self._record_has_report(idx):
                    self.report_indices_by_modality["mri"].append(idx)
                continue

            if mod_name in {"xray", "x-ray", "cxr", "chest xray", "chest x-ray"}:
                self.xray_indices.append(idx)
                if self._record_has_report(idx):
                    self.report_indices_by_modality["xray"].append(idx)
            elif mod_name == "ct":
                self.ct_indices.append(idx)
                if self._record_has_report(idx):
                    self.report_indices_by_modality["ct"].append(idx)

        # MRI ROI class-balanced pools.
        self.mri_roi_cls_by_class = {0: [], 1: [], 2: []}
        for idx in self.mri_roi_cls_indices:
            ann = self.records[idx].get("record", {}).get("annotations", {}) or {}
            cls_id = ann.get("classification_label", None)
            try:
                cls_id = int(cls_id)
            except (TypeError, ValueError):
                continue
            if cls_id in self.mri_roi_cls_by_class:
                self.mri_roi_cls_by_class[cls_id].append(idx)

        required_named_pools = {
            "MRI full-image": self.mri_full_indices,
            "MRI segmentation": self.mri_seg_indices,
            "MRI ROI classification": self.mri_roi_cls_indices,
            "X-ray": self.xray_indices,
            "CT": self.ct_indices,
            "BUS-BRA": self.busbra_all_indices,
        }
        for name, pool in required_named_pools.items():
            if len(pool) == 0:
                raise RuntimeError(f"No {name} samples found.")

        if not any(self.busbra_birads_by_class.values()):
            raise RuntimeError("No labelled BUS-BRA BI-RADS samples found.")
        if not any(self.busbra_pathology_by_class.values()):
            raise RuntimeError("No labelled BUS-BRA pathology samples found.")

        self.report_modalities = [
            name
            for name in ("xray", "ct", "mri", "ultrasound")
            if self.report_indices_by_modality[name]
        ]
        if not self.report_modalities:
            raise RuntimeError(
                "No report-supervised samples were found. Full report training "
                "requires at least one valid report in the training split."
            )

        for k, v in self.mri_roi_cls_by_class.items():
            if len(v) == 0:
                print(f"[Warn] No MRI ROI cls samples for class {k}")
        for k, v in self.busbra_birads_by_class.items():
            if len(v) == 0:
                print(f"[Warn] No BUS-BRA BI-RADS samples for local class {k}")
        for k, v in self.busbra_pathology_by_class.items():
            if len(v) == 0:
                print(f"[Warn] No BUS-BRA pathology samples for class {k}")

        self._reset_work_pools()

        print("[EqualModalityBatchSampler / MoFX-Dense]")
        print(f"  MRI full              : {len(self.mri_full_indices)}")
        print(f"  MRI seg               : {len(self.mri_seg_indices)}")
        print(f"  MRI ROI cls           : {len(self.mri_roi_cls_indices)}")
        print(f"  MRI ROI cls by class  : { {k: len(v) for k, v in self.mri_roi_cls_by_class.items()} }")
        print(f"  X-ray                 : {len(self.xray_indices)}")
        print(f"  CT                    : {len(self.ct_indices)}")
        print(f"  BUS-BRA all           : {len(self.busbra_all_indices)}")
        print(f"  BUS-BRA BI-RADS pools : { {k: len(v) for k, v in self.busbra_birads_by_class.items()} }")
        print(f"  BUS-BRA path pools    : { {k: len(v) for k, v in self.busbra_pathology_by_class.items()} }")
        print("  composition            : X-ray=2, CT=2, MRI=2, US=2")
        print("  MRI sub-pools          : rotating full/seg/ROI pairs")
        print("  US protected slots     : BI-RADS-balanced + pathology-balanced")
        print(
            "  report supervision     : at least one labelled report every batch "
            f"(available modalities: {', '.join(self.report_modalities)})"
        )
        print(f"  batch_size            : {self.batch_size}")
        print(f"  batches/epoch         : {self.batches_per_epoch}")

    def set_epoch(self, epoch):
        """Make shuffling reproducible while changing it between epochs."""
        self.epoch = int(epoch)
        self.rng.seed(self.base_seed + self.epoch)

    def _shuffle_copy(self, xs):
        xs = list(xs)
        if self.shuffle:
            self.rng.shuffle(xs)
        return xs

    def _record_key(self, idx):
        rec = self.records[int(idx)]
        return os.path.normcase(os.path.normpath(str(rec.get("img_path", idx))))

    def _record_has_report(self, idx):
        """Return whether a record carries usable report supervision."""
        rec = self.records[int(idx)]
        raw_flag = rec.get("has_rep", None)
        if raw_flag is not None:
            if hasattr(raw_flag, "item"):
                raw_flag = raw_flag.item()
            if bool(raw_flag):
                return True
        return any(
            "report" in str(task).strip().lower()
            for task in rec.get("tasks", [])
        )

    def _record_modality(self, idx):
        rec = self.records[int(idx)]
        ds_name = str(rec.get("ds_name", "")).strip()
        if ds_name == "BUS_BRA":
            return "ultrasound"
        if ds_name.startswith("LSS_MRI"):
            return "mri"
        mod_name = str(rec.get("mod_name", "")).strip().lower()
        if mod_name in {"xray", "x-ray", "cxr", "chest xray", "chest x-ray"}:
            return "xray"
        return mod_name

    def _reset_work_pools(self):
        self._mri_full_work = self._shuffle_copy(self.mri_full_indices)
        self._mri_seg_work = self._shuffle_copy(self.mri_seg_indices)
        self._mri_roi_cls_work = self._shuffle_copy(self.mri_roi_cls_indices)
        self._xray_work = self._shuffle_copy(self.xray_indices)
        self._ct_work = self._shuffle_copy(self.ct_indices)

        self._mri_roi_cls_work_by_class = {
            k: self._shuffle_copy(v) for k, v in self.mri_roi_cls_by_class.items()
        }
        self._busbra_birads_work_by_class = {
            k: self._shuffle_copy(v) for k, v in self.busbra_birads_by_class.items()
        }
        self._busbra_pathology_work_by_class = {
            k: self._shuffle_copy(v) for k, v in self.busbra_pathology_by_class.items()
        }
        self._report_work_by_modality = {
            name: self._shuffle_copy(indices)
            for name, indices in self.report_indices_by_modality.items()
        }

        self._mri_roi_cycle = [0, 1, 2]
        self._birads_cycle = [0, 1, 2, 3]
        self._pathology_cycle = [0, 1]
        self._mri_pair_cycle = [
            ("full", "seg"),
            ("full", "roi"),
            ("seg", "roi"),
        ]
        if self.shuffle:
            self.rng.shuffle(self._mri_roi_cycle)
            self.rng.shuffle(self._birads_cycle)
            self.rng.shuffle(self._pathology_cycle)
            self.rng.shuffle(self._mri_pair_cycle)
        self._mri_roi_cycle_ptr = 0
        self._birads_cycle_ptr = 0
        self._pathology_cycle_ptr = 0
        self._mri_pair_cycle_ptr = 0
        self._report_modality_cycle = list(self.report_modalities)
        if self.shuffle:
            self.rng.shuffle(self._report_modality_cycle)
        self._report_modality_cycle_ptr = 0

    def _pop_with_refresh(self, work_attr, base_list):
        if len(base_list) == 0:
            raise RuntimeError(f"Cannot sample from empty pool: {work_attr}")
        work = getattr(self, work_attr)
        if len(work) == 0:
            work = self._shuffle_copy(base_list)
            setattr(self, work_attr, work)
        return work.pop()

    def _pop_unique_with_refresh(
        self, work_attr, base_list, used_keys, pool_name="modality"
    ):
        """Pop an image-unique item; allow duplication only if unavoidable."""
        fallback = None
        for _ in range(max(1, len(base_list))):
            idx = self._pop_with_refresh(work_attr, base_list)
            fallback = idx
            key = self._record_key(idx)
            if key not in used_keys:
                used_keys.add(key)
                return idx

        if fallback is None:
            raise RuntimeError(f"Unable to sample from {work_attr}")
        if pool_name not in self._duplicate_warning_pools:
            print(
                f"[Warn] {pool_name} image duplication within a batch was "
                "unavoidable; check that at least two distinct images are available."
            )
            self._duplicate_warning_pools.add(pool_name)
        used_keys.add(self._record_key(fallback))
        return fallback

    def _pop_class_unique(self, class_id, base_map, work_map, used_keys):
        base_list = base_map.get(class_id, [])
        if len(base_list) == 0:
            return None
        for _ in range(len(base_list)):
            work = work_map.get(class_id, [])
            if len(work) == 0:
                work = self._shuffle_copy(base_list)
                work_map[class_id] = work
            idx = work.pop()
            key = self._record_key(idx)
            if key not in used_keys:
                used_keys.add(key)
                return idx
        return None

    def _pop_balanced_unique(
        self,
        cycle,
        ptr_attr,
        base_map,
        work_map,
        fallback_work_attr,
        fallback_base,
        used_keys,
        pool_name,
    ):
        # Round-robin class choice; skip missing or temporarily duplicate-only
        # classes, then use the full labelled pool as a safe fallback.
        for _ in range(len(cycle)):
            ptr = getattr(self, ptr_attr)
            class_id = cycle[ptr % len(cycle)]
            setattr(self, ptr_attr, ptr + 1)
            idx = self._pop_class_unique(class_id, base_map, work_map, used_keys)
            if idx is not None:
                return idx
        return self._pop_unique_with_refresh(
            fallback_work_attr, fallback_base, used_keys, pool_name
        )

    def _pop_mri_roi_cls_balanced(self):
        # MRI has no overlap with BUS-BRA, so uniqueness tracking is unnecessary.
        for _ in range(len(self._mri_roi_cycle)):
            class_id = self._mri_roi_cycle[
                self._mri_roi_cycle_ptr % len(self._mri_roi_cycle)
            ]
            self._mri_roi_cycle_ptr += 1
            base_list = self.mri_roi_cls_by_class.get(class_id, [])
            if len(base_list) == 0:
                continue
            work = self._mri_roi_cls_work_by_class[class_id]
            if len(work) == 0:
                work = self._shuffle_copy(base_list)
                self._mri_roi_cls_work_by_class[class_id] = work
            return work.pop()
        return self._pop_with_refresh(
            "_mri_roi_cls_work", self.mri_roi_cls_indices
        )

    def _pop_mri_pair(self):
        pair = self._mri_pair_cycle[
            self._mri_pair_cycle_ptr % len(self._mri_pair_cycle)
        ]
        self._mri_pair_cycle_ptr += 1

        def draw(pool_name):
            if pool_name == "full":
                return self._pop_with_refresh(
                    "_mri_full_work", self.mri_full_indices
                )
            if pool_name == "seg":
                return self._pop_with_refresh(
                    "_mri_seg_work", self.mri_seg_indices
                )
            if pool_name == "roi":
                return self._pop_mri_roi_cls_balanced()
            raise RuntimeError(f"Unknown MRI sampling pool: {pool_name}")

        return [draw(pool_name) for pool_name in pair]

    def _pop_report_unique(self, modality, used_keys):
        base = self.report_indices_by_modality[modality]
        work = self._report_work_by_modality[modality]
        fallback = None
        for _ in range(max(1, len(base))):
            if not work:
                work = self._shuffle_copy(base)
                self._report_work_by_modality[modality] = work
            idx = work.pop()
            fallback = idx
            if self._record_key(idx) not in used_keys:
                return idx
        return fallback

    def _ensure_report_supervision(self, batch):
        """Preserve modality counts while guaranteeing report loss each step."""
        if any(self._record_has_report(idx) for idx in batch):
            return batch

        for _ in range(len(self._report_modality_cycle)):
            modality = self._report_modality_cycle[
                self._report_modality_cycle_ptr
                % len(self._report_modality_cycle)
            ]
            self._report_modality_cycle_ptr += 1
            positions = [
                pos
                for pos, idx in enumerate(batch)
                if self._record_modality(idx) == modality
            ]
            if not positions:
                continue
            used_keys = {
                self._record_key(idx)
                for idx in batch
                if self._record_modality(idx) == modality
            }
            report_idx = self._pop_report_unique(modality, used_keys)
            if report_idx is None:
                continue
            batch[positions[-1]] = report_idx
            break

        if not any(self._record_has_report(idx) for idx in batch):
            raise RuntimeError(
                "Unable to construct a modality-balanced batch with report "
                "supervision. Check the training annotations."
            )
        return batch

    def __iter__(self):
        self._reset_work_pools()

        birads_labelled = [
            idx for pool in self.busbra_birads_by_class.values() for idx in pool
        ]
        pathology_labelled = [
            idx for pool in self.busbra_pathology_by_class.values() for idx in pool
        ]
        self._busbra_birads_fallback_work = self._shuffle_copy(birads_labelled)
        self._busbra_pathology_fallback_work = self._shuffle_copy(pathology_labelled)

        for _ in range(self.batches_per_epoch):
            used_busbra_images = set()

            busbra_birads_idx = self._pop_balanced_unique(
                self._birads_cycle,
                "_birads_cycle_ptr",
                self.busbra_birads_by_class,
                self._busbra_birads_work_by_class,
                "_busbra_birads_fallback_work",
                birads_labelled,
                used_busbra_images,
                "BUS-BRA",
            )
            busbra_pathology_idx = self._pop_balanced_unique(
                self._pathology_cycle,
                "_pathology_cycle_ptr",
                self.busbra_pathology_by_class,
                self._busbra_pathology_work_by_class,
                "_busbra_pathology_fallback_work",
                pathology_labelled,
                used_busbra_images,
                "BUS-BRA",
            )

            xray_used = set()
            xray_pair = [
                self._pop_unique_with_refresh(
                    "_xray_work", self.xray_indices, xray_used, "X-ray"
                )
                for _ in range(2)
            ]
            ct_used = set()
            ct_pair = [
                self._pop_unique_with_refresh(
                    "_ct_work", self.ct_indices, ct_used, "CT"
                )
                for _ in range(2)
            ]
            mri_pair = self._pop_mri_pair()

            batch = [
                *xray_pair,
                *ct_pair,
                *mri_pair,
                busbra_birads_idx,
                busbra_pathology_idx,
            ]
            batch = self._ensure_report_supervision(batch)

            if self.shuffle:
                self.rng.shuffle(batch)

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and len(batch) > 0:
                yield batch

    def __len__(self):
        return self.batches_per_epoch

def create_optimizer(
    model,
    lr_backbone=5e-6,
    lr_moe=2e-4,
    lr_moe_gates=2e-4,
    lr_seg=5e-4,
    lr_cls=2e-4,
    lr_report=1e-4,
    lr_detector=2e-4,
    lr_rest=2e-4,
    weight_decay=0.05,
):
    """Create paper-aligned optimizer parameter groups for MoFX-Dense.

    Router/context/gate/scalar parameters use the MoFX learning rate and every
    trainable group uses the manuscript's AdamW weight decay.
    """
    SEG_KEYS = ("seg_decoder", "seg_report_token_extractor")
    CLS_KEYS = (
        "attn_pool", "roi_pool", "global_pool", "cls_fuse", "cls_neck",
        "cls_head", "birads_head", "pathology_meas_head",
        "pathology_visual_neck", "ultrasound_adapter",
        "classification_gradif_fusion_gate",
        "measurement_encoder", "mask_roi_pool",
    )
    REPORT_KEYS = ("visual_adapter", "report_decoder", "lora")

    group_names = (
        "backbone", "moe", "moe_gates", "seg", "cls",
        "report", "detector", "rest",
    )
    groups = {k: [] for k in group_names}

    gate_markers = (
        ".raw_gamma", ".raw_expert_scale",
        ".raw_task_prior_residual", ".raw_modality_task_prior_residual",
        ".router.", ".gate_mlp.",
        ".router_context_encoder.", ".expert_keys",
        ".raw_router_logit_scale",
        ".route_to_content.", ".modality_to_content.",
        ".route_to_energy.", ".modality_to_energy.",
        ".context_interaction_proj.", ".spatial_proj.",
        ".content_proj.", ".energy_proj.",
        ".task_embedding.", ".route_embedding.", ".modality_embedding.",
        ".task_router_bias.", ".output_router_bias",
        ".route_film_scale.", ".route_film_shift.",
        ".alpha_head.", ".output_alpha_bias",
        ".reliability_head.", ".gamma_head.", ".route_gamma_bias",
        ".route_prior_logits", ".modality_route_prior_logits",
        ".band_decomposition.",
    )

    for name, param in model.named_parameters():
        if name.startswith("swin."):
            groups["backbone"].append(param)
        elif name.startswith("task_moe."):
            if any(marker in name for marker in gate_markers):
                groups["moe_gates"].append(param)
            else:
                groups["moe"].append(param)
        elif name in ("segmentation_gradif_fusion_logits",):
            groups["moe_gates"].append(param)
        elif name.startswith("pathology_identity_gate."):
            groups["moe_gates"].append(param)
        elif name.startswith("detector."):
            groups["detector"].append(param)
        elif any(k in name for k in SEG_KEYS):
            groups["seg"].append(param)
        elif any(k in name for k in CLS_KEYS):
            groups["cls"].append(param)
        elif any(k in name for k in REPORT_KEYS):
            groups["report"].append(param)
        else:
            groups["rest"].append(param)

    lr_map = {
        "backbone": lr_backbone,
        "moe": lr_moe,
        "moe_gates": lr_moe_gates,
        "seg": lr_seg,
        "cls": lr_cls,
        "report": lr_report,
        "detector": lr_detector,
        "rest": lr_rest,
    }
    param_groups = []
    for name in group_names:
        param_groups.append({
            "name": name,
            "params": groups[name],
            "lr": float(lr_map[name]),
            "weight_decay": float(weight_decay),
        })

    print("[Optimizer] param groups:")
    for pg in param_groups:
        n_trainable = sum(1 for p in pg["params"] if p.requires_grad)
        print(
            f"  {pg['name']:<9} n={len(pg['params']):<4} "
            f"trainable={n_trainable:<4} lr={pg['lr']:.2e} "
            f"wd={pg['weight_decay']:.2e}"
        )

    return optim.AdamW(
        param_groups,
        fused=torch.cuda.is_available(),
    )

def print_teacher_map(num_modalities, num_tasks, class_names_map=None):
    """
    Prints the Dynamic Teacher Assignment Map so you can verify the logic.
    Formula: Target_Expert = (Modality_ID * Num_Tasks) + Task_ID
    """
    
    # Default names if none provided
    class_names_map = {
        0: 'CT', 
        1: 'MRI', 
        2: 'Ultrasound', 
        3: 'XRay'
    }
    
    task_names = {0: "Segmentation", 1: "Classification", 2: "Reporting", 3: "Detection"}

    print("\n" + "="*60)
    print(f"TEACHER FORCING ASSIGNMENT MAP (Curriculum Goal)")
    print("="*60)
    print(f"{'Modality ID':<5} | {'Modality Name':<15} | {'Task ID':<5} | {'Task Name':<15} | {'TARGET EXPERT':<15}")
    print("-"*60)

    for m_id in range(num_modalities):
        for t_id in range(num_tasks):
            
            # THE DYNAMIC FORMULA
            target_expert = (m_id * num_tasks) + t_id
            
            m_name = class_names_map.get(m_id, f"Mod_{m_id}")
            t_name = task_names.get(t_id, f"Task_{t_id}")
            
            print(f"{m_id:<5} | {m_name:<15} | {t_id:<5} | {t_name:<15} | E{target_expert:<13}")
    
    print("="*60 + "\n")

class AdaptiveFocalLoss(nn.Module):
    """Focal BCE with:
    - alpha=0.5  (neutral; old 0.25 over-suppressed rare positive classes)
    - pos_weight support (class-imbalance weighting inside BCE term)
    - label_smoothing (default 0.05): prevents Sensitivity=0 collapse on
      rare classes by stopping the model driving logits to +/-inf
    """
    def __init__(self, alpha=0.5, gamma=2.0, label_smoothing=0.05):   # alpha 0.25 -> 0.5 (neutral)
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets, pos_weight=None):
        # Label smoothing: 0 -> eps, 1 -> (1-eps)
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets_s = targets * (1.0 - eps) + (1.0 - targets) * eps
        else:
            targets_s = targets

        bce = F.binary_cross_entropy_with_logits(
            logits, targets_s, reduction='none',
            pos_weight=pos_weight,        # <-- now actually used
        )
        probs = torch.sigmoid(logits)
        # Focal weight against unsmoothed targets (correct p_t definition)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return alpha_t * focal_weight * bce  # unreduced [B, C]
    
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, targets, focal_loss=None, pos_weight=None):
        eps = self.smoothing
        targets_smooth = targets * (1.0 - eps) + (1.0 - targets) * eps

        if focal_loss is not None:
            return focal_loss(logits, targets_smooth)  # no reduction
        else:
            return F.binary_cross_entropy_with_logits(
                logits,
                targets_smooth,
                pos_weight=pos_weight,
                reduction='none'   # 🔥 IMPORTANT
            )

def set_group_lrs(optimizer, **lrs):
    """
    Accept either 'backbone' or 'lr_backbone' as kwarg names.
    Prints the resulting LRs so you can verify it actually fired.
    """
    lrs = {(k[3:] if k.startswith("lr_") else k): v for k, v in lrs.items()}
    changed = []
    for g in optimizer.param_groups:
        n = g.get("name", "")
        if n in lrs and lrs[n] is not None:
            g["lr"] = float(lrs[n])
            changed.append((n, g["lr"]))
    if changed:
        print("[set_group_lrs] applied:", ", ".join(f"{n}={lr:.2e}" for n, lr in changed))
    else:
        print(f"[set_group_lrs] WARNING: nothing matched. kwargs={list(lrs.keys())}")

def normalize_boxes(boxes, img_w, img_h):
    boxes = boxes.clone().float()
    boxes[:, 0::2] /= img_w
    boxes[:, 1::2] /= img_h
    return box_xyxy_to_cxcywh(boxes)

def show_det_debug(
    img_tensor,
    pred_boxes,
    pred_labels,
    pred_scores,
    gt_boxes,
    gt_labels,
    det_class_names=None,
    title="Detection Debug",
    save_path=None,
    show_window=True
):
    try:
        img = img_tensor.detach().cpu()

        # CHW -> HWC
        if img.ndim == 3:
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()

        img = img.astype(np.float32)

        # Normalize for display
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        # Handle single-channel image shapes cleanly
        use_gray = False
        if img.ndim == 2:
            use_gray = True
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = img[..., 0]
            use_gray = True

        if show_window:
            plt.ion()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), num="det_debug")
        fig.clf()
        axes = fig.subplots(1, 2)

        # --------------------------------------------------
        # Left: GT
        # --------------------------------------------------
        ax = axes[0]
        if use_gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title("GT Detection")
        ax.axis("off")

        if gt_boxes is not None and gt_boxes.numel() > 0:
            gt_boxes_np = gt_boxes.detach().cpu().numpy()
            gt_labels_np = gt_labels.detach().cpu().numpy() if gt_labels is not None and gt_labels.numel() > 0 else None

            for j, box in enumerate(gt_boxes_np):
                x1, y1, x2, y2 = box.tolist()
                rect = plt.Rectangle(
                    (x1, y1),
                    max(1.0, x2 - x1),
                    max(1.0, y2 - y1),
                    fill=False,
                    edgecolor='lime',
                    linewidth=2
                )
                ax.add_patch(rect)

                if gt_labels_np is not None:
                    lbl = int(gt_labels_np[j])
                    lbl_txt = (
                        det_class_names[lbl]
                        if det_class_names is not None and 0 <= lbl < len(det_class_names)
                        else str(lbl)
                    )
                    ax.text(
                        x1,
                        max(0, y1 - 4),
                        f"GT:{lbl_txt}",
                        color='lime',
                        fontsize=9,
                        bbox=dict(facecolor='black', alpha=0.5, pad=1)
                    )

        # --------------------------------------------------
        # Right: Predictions
        # --------------------------------------------------
        ax = axes[1]
        if use_gray:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title("Pred Detection")
        ax.axis("off")

        if pred_boxes is not None and pred_boxes.numel() > 0:
            pred_boxes_np = pred_boxes.detach().cpu().numpy()
            pred_labels_np = pred_labels.detach().cpu().numpy() if pred_labels is not None and pred_labels.numel() > 0 else None
            pred_scores_np = pred_scores.detach().cpu().numpy() if pred_scores is not None and pred_scores.numel() > 0 else None

            for j, box in enumerate(pred_boxes_np):
                x1, y1, x2, y2 = box.tolist()
                rect = plt.Rectangle(
                    (x1, y1),
                    max(1.0, x2 - x1),
                    max(1.0, y2 - y1),
                    fill=False,
                    edgecolor='red',
                    linewidth=2
                )
                ax.add_patch(rect)

                txt = "Pred"
                if pred_labels_np is not None:
                    lbl = int(pred_labels_np[j])
                    lbl_txt = (
                        det_class_names[lbl]
                        if det_class_names is not None and 0 <= lbl < len(det_class_names)
                        else str(lbl)
                    )
                    txt += f":{lbl_txt}"
                if pred_scores_np is not None:
                    txt += f" {pred_scores_np[j]:.2f}"

                ax.text(
                    x1,
                    max(0, y1 - 4),
                    txt,
                    color='red',
                    fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1)
                )

        fig.suptitle(title)
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if show_window:
            plt.show(block=False)
            plt.pause(0.001)
        else:
            plt.close(fig)

    except Exception as e:
        print(f"[DET DEBUG SHOW ERROR] {e}")

def prepare_detection_targets(det_targets, active_mask, img_h, img_w, num_classes, device):
    active_idx = active_mask.nonzero(as_tuple=True)[0]
    out = []

    for b in active_idx.tolist():
        t = det_targets[b]

        if t is None:
            boxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            labels = torch.zeros((0,), dtype=torch.long, device=device)
        else:
            if "boxes_norm" in t and t["boxes_norm"] is not None:
                boxes = t["boxes_norm"].to(device=device, dtype=torch.float32)
            else:
                boxes = normalize_boxes(
                    t["boxes"].to(device=device, dtype=torch.float32),
                    img_w,
                    img_h,
                )

            labels = t["labels"].to(device=device, dtype=torch.long)

        assert boxes.ndim == 2 and boxes.shape[-1] == 4, f"bad boxes shape: {boxes.shape}"
        assert labels.ndim == 1, f"bad labels shape: {labels.shape}"
        assert labels.shape[0] == boxes.shape[0], f"label/box count mismatch: {labels.shape[0]} vs {boxes.shape[0]}"

        if boxes.numel() > 0:
            assert torch.isfinite(boxes).all(), "NaN/Inf in boxes"
            assert boxes.min() >= -1e-6 and boxes.max() <= 1.0 + 1e-6, \
                f"boxes not normalized cxcywh in [0,1], min={boxes.min().item()}, max={boxes.max().item()}"

        if labels.numel() > 0:
            assert ((labels >= 0) & (labels < num_classes)).all(), \
                f"labels out of range: {labels.tolist()} num_classes={num_classes}"

        out.append({"boxes": boxes, "labels": labels})

    return out, active_idx

def build_mri_roi_targets_from_matches(
    roi_batch_idx,
    roi_box_idx,
    det_match_indices,
    active_to_batch_idx,
    det_targets
):
    """
    roi_batch_idx: [Nroi] original batch indices
    roi_box_idx:   [Nroi] predicted query indices OR GT indices depending on roi source
    det_match_indices: list of (pred_ids, gt_ids) for detection-active samples
    active_to_batch_idx: maps detection-active index -> original batch index
    """
    matched_targets = []

    # build lookup:
    # (orig_batch_idx, pred_idx) -> gt_idx
    pred_to_gt = {}
    for active_i, (pred_ids, gt_ids) in enumerate(det_match_indices):
        orig_b = active_to_batch_idx[active_i]
        for p, g in zip(pred_ids.tolist(), gt_ids.tolist()):
            pred_to_gt[(orig_b, p)] = g

    for k in range(len(roi_batch_idx)):
        b = int(roi_batch_idx[k].item())
        pred_j = int(roi_box_idx[k].item())

        if (b, pred_j) not in pred_to_gt:
            return None

        gt_j = pred_to_gt[(b, pred_j)]

        if 'severity' not in det_targets[b]:
            return None
        if gt_j >= det_targets[b]['severity'].numel():
            return None

        matched_targets.append(det_targets[b]['severity'][gt_j])

    if len(matched_targets) == 0:
        return None

    return torch.stack(matched_targets).long()

def build_gt_roi_query_ids_from_matches(det_match_indices, active_to_batch_idx, det_targets):
    """
    Builds per-image mapping:
        gt_box_index -> matched predicted query index

    returns:
        roi_query_ids: list of length B
            roi_query_ids[b] is Tensor [num_gt_boxes_b] filled with:
                matched pred query index, or -1 if unmatched
    """
    roi_query_ids = []

    for b in range(len(det_targets)):
        num_gt = det_targets[b]["boxes"].shape[0] if "boxes" in det_targets[b] else 0
        roi_query_ids.append(
            torch.full(
                (num_gt,),
                -1,
                dtype=torch.long,
                device=det_targets[b]["boxes"].device if num_gt > 0 else det_targets[0]["boxes"].device
            )
        )

    for active_i, (pred_ids, gt_ids) in enumerate(det_match_indices):
        orig_b = active_to_batch_idx[active_i]

        if roi_query_ids[orig_b].numel() == 0:
            continue

        for p, g in zip(pred_ids.tolist(), gt_ids.tolist()):
            if g < roi_query_ids[orig_b].numel():
                roi_query_ids[orig_b][g] = p

    return roi_query_ids

def build_detection_criterion(det_num_classes, device, num_decoder_layers=6):
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_bbox=5.0,
        cost_giou=2.0
    )

    weight_dict = {
        "loss_ce": 1.0,      # closer to official DINO
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_dn_ce": 0.5,   # keep your DN weights unless you port official criterion fully
        "loss_dn_bbox": 2.0,
        "loss_dn_giou": 1.0,
    }

    for i in range(num_decoder_layers - 1):
        weight_dict.update({
            f"loss_ce_{i}": 1.0,
            f"loss_bbox_{i}": 5.0,
            f"loss_giou_{i}": 2.0,
        })

    crit = DINOCriterion(
        num_classes=det_num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        use_focal=True,
        alpha=0.25,
        gamma=2.0
    ).to(device, non_blocking=True)
    return crit

def estimate_mri_severity_class_weights(loader, num_classes=4, max_batches=None):
    counts = torch.ones(num_classes, dtype=torch.float64)  # smoothing

    for bi, batch in enumerate(loader):
        det_targets = batch.get('det_targets', None)
        if det_targets is None:
            continue

        for sample_idx, t in enumerate(det_targets):
            if 'detection' not in str(batch['tasks'][sample_idx]).lower():
                continue
            if 'severity' not in t:
                continue

            sev = t['severity']
            if not torch.is_tensor(sev):
                sev = torch.as_tensor(sev)

            sev = sev.view(-1).long()
            sev = sev[(sev >= 0) & (sev < num_classes)]
            if sev.numel() == 0:
                continue

            counts += torch.bincount(sev, minlength=num_classes).double()

        if max_batches is not None and (bi + 1) >= max_batches:
            break

    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    return weights.float()

def move_multitask_batch_to_device(batch, device):
    imgs = batch['image'].to(device, non_blocking=True)
    mod_ids = batch['modality_id'].to(device, non_blocking=True)

    captions = batch.get('report_ids', None)
    report_labels = batch.get('report_labels', None)
    if captions is not None:
        captions = captions.to(device, non_blocking=True)
    if report_labels is not None:
        report_labels = report_labels.to(device, non_blocking=True)

    task_mask = batch['task_mask'].to(device, non_blocking=True)
    has_seg = batch['has_seg'].to(device, non_blocking=True)
    has_cls = batch['has_cls'].to(device, non_blocking=True)
    has_rep = batch['has_rep'].to(device, non_blocking=True)
    has_det = batch['has_det'].to(device, non_blocking=True)
    has_birads = batch.get('has_birads', torch.zeros_like(has_cls)).to(
        device, non_blocking=True
    )
    has_pathology = batch.get('has_pathology', torch.zeros_like(has_cls)).to(
        device, non_blocking=True
    )
    measurements = batch.get('measurements', None)
    measurements_present = batch.get('measurements_present', None)
    measurement_sizes = batch.get('measurement_sizes', None)
    busbra_seg_channel_indices = batch.get('busbra_seg_channel_indices', None)
    pathology_index = batch.get('pathology_index', None)
    birads_index = batch.get('birads_index', None)
    if measurements is not None:
        measurements = measurements.to(device, non_blocking=True)
    if measurements_present is not None:
        measurements_present = measurements_present.to(device, non_blocking=True)
    if pathology_index is not None:
        pathology_index = pathology_index.to(device, non_blocking=True)
    if birads_index is not None:
        birads_index = birads_index.to(device, non_blocking=True)

    is_lss_mri = batch.get('is_lss_mri', None)
    if is_lss_mri is not None:
        is_lss_mri = is_lss_mri.to(device, non_blocking=True)

    det_targets = batch.get('det_targets', None)
    if det_targets is not None:
        det_targets = [
            {
                'boxes': t['boxes'].to(device, non_blocking=True),
                'boxes_norm': t['boxes_norm'].to(device, non_blocking=True) if 'boxes_norm' in t else None,
                'labels': t['labels'].to(device, non_blocking=True),
                **({'severity': t['severity'].to(device, non_blocking=True)} if 'severity' in t else {})
            }
            for t in det_targets
        ]

    gt_boxes_for_roi = batch.get('gt_boxes_for_roi', None)
    if gt_boxes_for_roi is not None:
        gt_boxes_for_roi = [b.to(device, non_blocking=True) for b in gt_boxes_for_roi]

    return {
        "imgs": imgs,
        "mod_ids": mod_ids,
        "captions": captions,
        "report_labels": report_labels,
        "task_mask": task_mask,
        "has_seg": has_seg,
        "has_cls": has_cls,
        "has_rep": has_rep,
        "has_det": has_det,
        "has_birads": has_birads,
        "has_pathology": has_pathology,
        "measurements": measurements,
        "measurements_present": measurements_present,
        "measurement_sizes": measurement_sizes,
        "busbra_seg_channel_indices": busbra_seg_channel_indices,
        "pathology_index": pathology_index,
        "birads_index": birads_index,
        "is_lss_mri": is_lss_mri,
        "det_targets": det_targets,
        "gt_boxes_for_roi": gt_boxes_for_roi,
    }

def inspect_one_det_sample(ds, idx):
    s = ds[idx]
    assert s["has_det"].item() and s["is_lss_mri"].item(), f"sample {idx} is not MRI+detection"

    t = s["det_targets"]
    print("=" * 80)
    print(f"[DET SAMPLE CHECK] idx={idx}")
    print("path:", s["image_path"])
    print("ds_name:", s["ds_name"])
    print("tasks:", s["tasks"])
    print("img shape:", tuple(s["image"].shape))

    print("boxes abs shape:", tuple(t["boxes"].shape))
    print("boxes abs:\n", t["boxes"])
    print("boxes norm shape:", tuple(t["boxes_norm"].shape))
    print("boxes norm:\n", t["boxes_norm"])
    print("labels:", t["labels"].tolist())

    if "severity" in t:
        print("severity:", t["severity"].tolist())

    assert t["boxes"].ndim == 2 and t["boxes"].shape[1] == 4, f"bad abs box shape: {t['boxes'].shape}"
    assert t["boxes_norm"].ndim == 2 and t["boxes_norm"].shape[1] == 4, f"bad norm box shape: {t['boxes_norm'].shape}"
    assert t["labels"].shape[0] == t["boxes"].shape[0], "label/box count mismatch"
    assert t["boxes_norm"].shape[0] == t["boxes"].shape[0], "norm-box/box count mismatch"

    if t["boxes"].numel() > 0:
        x1, y1, x2, y2 = t["boxes"].unbind(-1)

        assert torch.isfinite(t["boxes"]).all(), "NaN/Inf in abs boxes"
        assert torch.isfinite(t["boxes_norm"]).all(), "NaN/Inf in norm boxes"

        assert (x2 > x1).all(), "invalid x box"
        assert (y2 > y1).all(), "invalid y box"

        assert x1.min() >= 0, f"x1 min < 0: {x1.min().item()}"
        assert y1.min() >= 0, f"y1 min < 0: {y1.min().item()}"
        assert x2.max() <= ds.target_size[0] + 1e-3, f"x2 max too large: {x2.max().item()}"
        assert y2.max() <= ds.target_size[1] + 1e-3, f"y2 max too large: {y2.max().item()}"

        bnorm = t["boxes_norm"]
        assert bnorm.min() >= -1e-6, f"normalized box min too small: {bnorm.min().item()}"
        assert bnorm.max() <= 1.0 + 1e-6, f"normalized box max too large: {bnorm.max().item()}"

        # Optional: decode normalized cxcywh back to abs xyxy and compare
        cx, cy, w, h = bnorm.unbind(-1)
        x1_dec = (cx - 0.5 * w) * ds.target_size[0]
        y1_dec = (cy - 0.5 * h) * ds.target_size[1]
        x2_dec = (cx + 0.5 * w) * ds.target_size[0]
        y2_dec = (cy + 0.5 * h) * ds.target_size[1]
        decoded = torch.stack([x1_dec, y1_dec, x2_dec, y2_dec], dim=-1)

        max_abs_err = (decoded - t["boxes"]).abs().max().item()
        print("max decode(abs(norm_box) - abs_box) error:", max_abs_err)

    print("[OK] sample passed checks")


def visualize_one_det_sample(ds, idx, unnormalize=True):
    s = ds[idx]
    assert s["has_det"].item() and s["is_lss_mri"].item(), f"sample {idx} is not MRI+detection"

    img = s["image"].detach().cpu().clone()   # [3,H,W]
    t = s["det_targets"]

    if unnormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean

    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    ax.set_title(f"idx={idx}\n{s['image_path']}")
    ax.axis("off")

    boxes = t["boxes"].cpu()
    labels = t["labels"].cpu().tolist()

    for j, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="lime",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(0, y1 - 3),
            f"{labels[j]}",
            color="yellow",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=1)
        )

    plt.tight_layout()
    plt.show()


def debug_detection_dataset(ds, num_samples=5, seed=123):
    det_indices = [
        i for i in range(len(ds))
        if ds.data[i]["has_det"].item() and ds.data[i]["is_lss_mri"].item()
    ]

    print(f"[DET DEBUG] total MRI detection samples in split: {len(det_indices)}")
    if len(det_indices) == 0:
        print("[DET DEBUG] No MRI detection samples found.")
        return

    random.seed(seed)
    chosen = det_indices[:min(num_samples, len(det_indices))]
    if len(det_indices) > num_samples:
        chosen = random.sample(det_indices, num_samples)

    print("[DET DEBUG] chosen indices:", chosen)

    for idx in chosen:
        inspect_one_det_sample(ds, idx)
        visualize_one_det_sample(ds, idx)

@torch.no_grad()
def compute_local_multiclass_weights(train_ds, ds_name, num_local_classes=3, sqrt=True):
    counts = np.zeros(num_local_classes, dtype=np.float32)

    for item in train_ds.data:
        if item['ds_name'] != ds_name:
            continue
        if not item['has_cls'].item():
            continue

        cls_idx = int(item.get('class_index', torch.tensor(-100)).item())
        if 0 <= cls_idx < num_local_classes:
            counts[cls_idx] += 1.0

    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    if sqrt:
        inv = np.sqrt(inv)

    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)

def debug_cxcywh_boxes(boxes, name, batch_paths=None):
    if boxes is None or boxes.numel() == 0:
        print(f"[{name}] empty")
        return

    boxes = boxes.detach()
    finite = torch.isfinite(boxes).all(dim=-1)
    nonneg_wh = (boxes[..., 2:] >= 0).all(dim=-1)

    bad = ~(finite & nonneg_wh)

    print(f"[{name}] shape={tuple(boxes.shape)} "
          f"finite_all={bool(finite.all().item())} "
          f"nonneg_wh_all={bool(nonneg_wh.all().item())}")

    if bad.any():
        bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
        print(f"[{name}] BAD indices: {bad_idx[:10]}")
        for i in bad_idx[:10]:
            print(f"  {name}[{i}] = {boxes[i].cpu().tolist()}")

def debug_xyxy_boxes(boxes, name):
    if boxes is None or boxes.numel() == 0:
        print(f"[{name}] empty")
        return

    boxes = boxes.detach()
    finite = torch.isfinite(boxes).all(dim=-1)
    valid_xyxy = (boxes[..., 2:] >= boxes[..., :2]).all(dim=-1)

    bad = ~(finite & valid_xyxy)

    print(f"[{name}] shape={tuple(boxes.shape)} "
          f"finite_all={bool(finite.all().item())} "
          f"valid_xyxy_all={bool(valid_xyxy.all().item())}")

    if bad.any():
        bad_idx = torch.nonzero(bad, as_tuple=False).view(-1).tolist()
        print(f"[{name}] BAD indices: {bad_idx[:10]}")
        for i in bad_idx[:10]:
            print(f"  {name}[{i}] = {boxes[i].cpu().tolist()}")

def parameter_count_summary(model):
    def count(params):
        return int(sum(p.numel() for p in params))
    task_moe = getattr(model, "task_moe", None)
    return {
        "total": count(model.parameters()),
        "trainable": count(p for p in model.parameters() if p.requires_grad),
        "task_moe_total": count(task_moe.parameters()) if task_moe is not None else 0,
        "task_moe_trainable": count(
            p for p in task_moe.parameters() if p.requires_grad
        ) if task_moe is not None else 0,
    }


def dataset_membership_fingerprint(dataset):
    """Hash stable sample identifiers for release provenance."""
    digest = hashlib.sha256()
    records = getattr(dataset, "data", [])
    preferred = (
        "ds_name", "dataset", "image_path", "image", "img_path",
        "case_id", "case", "patient_id", "id", "ID", "split",
    )
    for index, record in enumerate(records):
        if isinstance(record, dict):
            stable = {
                key: str(record[key])
                for key in preferred
                if key in record and record[key] is not None
            }
            if not stable:
                stable = {"index": index, "keys": sorted(record.keys())}
        else:
            stable = {"index": index, "value": str(record)}
        digest.update(json.dumps(stable, sort_keys=True).encode("utf-8"))
    return {
        "count": int(len(records)),
        "sha256": digest.hexdigest(),
    }


def run_pipeline(data_root, medclip_weights_path, dino_repo_path, dino_cfg_path, dino_pretrain_path, use_report_label_guidance,
                 epochs=100, debug=False, target_size=(224, 224), use_moe=True, resume=False, resume_run_id=None, ACCUM_STEPS=4,
                 learning_rates=None,
                 weight_decay=0.05,
                 scheduler_factor=0.5,
                 scheduler_patience=8,
                 scheduler_threshold=1e-4,
                 mixed_precision=True,
                 GRAD_CLIP_NORM=1.0,
                 # ----- MoE configuration -----
                 moe_type="gradif",
                 moe_num_experts=5,           # four directed FreqXperts + null
                 moe_top_k=2,                 # used only by the sparse baseline
                 gradif_attn_dim=96,
                 gradif_num_heads=4,
                 gradif_router_dim=128,
                 gradif_top_k=2,
                 gradif_routing_mode="dense_soft",
                 gradif_expert_bank_type="freqxpert",
                 gradif_temperature=1.5,
                 gradif_temperature_end=1.5,
                 gradif_temperature_hold_epochs=10,
                 gradif_temperature_anneal_epochs=10,
                 gradif_expert_warmup_epochs=5,
                 gradif_generic_mlp_bottleneck=96,
                 gradif_include_generalist=False,
                 gradif_include_null_expert=True,
                 gradif_modality_scale=1.0,
                 gradif_router_strategy="factorized_multigate",
                 gradif_router_task_residual_scale=0.10,
                 gradif_router_modality_residual_scale=0.10,
                 gradif_router_attention_pool=True,
                 gradif_router_alpha_content_init=0.50,
                 gradif_router_alpha_task_init=0.25,
                 gradif_router_alpha_modality_init=0.25,
                 gradif_router_task_specific_gates=True,
                 gradif_router_modality_specific_gates=True,
                 gradif_router_modality_content_adapters=True,
                 gradif_router_branch_hidden=96,
                 mofx_router_configuration="full",
                 gradif_router_mi_min_group_size=4,
                 gradif_router_mi_queue_size=32,
                 gradif_router_mi_current_fraction=0.50,
                 gradif_routing_mi_enabled=False,
                 gradif_routing_mi_weight_max=5e-4,
                 gradif_routing_mi_start_epoch=1,
                 gradif_routing_mi_ramp_end_epoch=3,
                 gradif_routing_mi_decay_start_epoch=6,
                 gradif_routing_mi_end_epoch=9,
                 gradif_use_granularity_prior=False,
                 gradif_decomposition="gaussian_fixed",  # manuscript Eq. (3): fixed Gaussian filters
                 gradif_fft_initial_boundaries=(0.25, 0.55),
                 gradif_fft_mask_temperature=0.08,
                 gradif_fft_min_bandwidth=0.05,
                 gradif_sparsity_weight=0.0,
                 gradif_load_balance_weight=0.0,
                 gradif_task_separation_weight=0.0,
                 gradif_expert_diversity_weight=0.0,
                 gradif_gate_weight=0.0,
                 gradif_router_conditioning_weight=0.0,  # no routing-shape penalty
                 gradif_logit_magnitude_weight=0.0,
                 gradif_regularization_warmup=6,
                 gradif_classification_base_aux_weight=0.0,
                 gradif_classification_no_harm_weight=0.0,
                 gradif_segmentation_no_harm_weight=0.0,
                 gradif_no_harm_start_epoch=8,
                 gradif_no_harm_ramp_epochs=6,
                 gradif_spatial_gate_identity_weight=0.0,
                 classification_gradif_fusion_init=0.25,
                 classification_gradif_max_delta_ratio=0.06,
                 descriptor_gt_warmup_epochs=0,
                 descriptor_gt_transition_epochs=0,
                 unified_batch_size=8,
                 modality_batch_counts=None,
                 batches_per_epoch=500,
                 # ----- BUS-BRA class-imbalance controls -----
                 busbra_use_class_weights=True,
                 birads_weight_power=0.40,
                 pathology_weight_power=0.50,
                 pathology_malignant_boost=1.15,
                 busbra_class_weight_min=0.50,
                 busbra_class_weight_max=3.00,
                 llm_model_name="meta-llama/Llama-3.2-1B-Instruct",
                 run_name="run",               # tags the results_<run_name>.json artifact
                 experiment_seed=42,
                 deterministic_run=True,
                 experiment_group_id="paper_mofx_dense",
                 run_base_dir="outputs/runs",
                 run_study_name="paper_release/final_mofx_dense",
                 gradif_active_stages=("c2", "c3", "c4_semantic"),
                  ):
    paper_lrs = {
        "backbone": 5e-6,
        "mofx": 2e-4,
        "segmentation": 5e-4,
        "classification": 2e-4,
        "report": 1e-4,
        "detector": 2e-4,
    }
    if learning_rates is not None:
        paper_lrs.update({key: float(value) for key, value in learning_rates.items()})
    set_reproducibility(seed=experiment_seed, deterministic=deterministic_run)
    final_architecture = {
        "use_moe": bool(use_moe),
        "moe_type": str(moe_type),
        "routing_mode": str(gradif_routing_mode),
        "expert_bank": str(gradif_expert_bank_type),
        "router": str(gradif_router_strategy),
        "decomposition": str(gradif_decomposition),
        "router_configuration": str(mofx_router_configuration),
    }
    required_architecture = {
        "use_moe": True,
        "moe_type": "gradif",
        "routing_mode": "dense_soft",
        "expert_bank": "freqxpert",
        "router": "factorized_multigate",
        "decomposition": "gaussian_fixed",
        "router_configuration": "full",
    }
    if final_architecture != required_architecture:
        raise ValueError(
            "This release contains only the final MoFX-Dense architecture; "
            f"received {final_architecture}."
        )
    print(
        f"[Unified MoE] bank={gradif_expert_bank_type}; "
        f"routing={gradif_routing_mode}; decomposition={gradif_decomposition}; "
        f"generalist={bool(gradif_include_generalist)}; "
        f"null={bool(gradif_include_null_expert)}; "
        f"temperature={float(gradif_temperature):.3f}->"
        f"{float(gradif_temperature_end):.3f}; "
        f"router={gradif_router_strategy}; "
        f"task_residual={float(gradif_router_task_residual_scale):.3f}; "
        f"modality_residual={float(gradif_router_modality_residual_scale):.3f}; "
        f"attention_pool={bool(gradif_router_attention_pool)}"
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if resume and resume_run_id is None:
        print("Error: You must provide 'resume_run_id' when resume=True")
        return
    manager = ExperimentManager(
        base_dir=str(run_base_dir), resume_run_id=resume_run_id
    )
    with open(
        os.path.join(manager.run_dir, "run_manifest.json"),
        "w", encoding="utf-8",
    ) as _manifest_handle:
        json.dump({
            "run_study_name": str(run_study_name),
            "run_base_dir": os.path.abspath(str(run_base_dir)),
            "active_stages": list(gradif_active_stages),
            "router_configuration": str(mofx_router_configuration),
            "decomposition": str(gradif_decomposition),
            "expert_bank_type": str(gradif_expert_bank_type),
            "include_null": bool(gradif_include_null_expert),
        }, _manifest_handle, indent=2)
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"[LLM] Using: {llm_model_name}")
    
    # ==========================================================
    # NORMALIZATION
    # ==========================================================
    # Same normalization used by the pretrained Swin/MedCLIP encoder.
    base_norm = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # ==========================================================
    # STANDARD DETERMINISTIC TRANSFORM
    # ==========================================================
    # This transform does not contain spatial augmentation.
    # BUS-BRA augmentation is performed jointly on image + mask
    # inside UnifiedMedicalDataset.__getitem__().
    transform_standard = transforms.Compose([
        transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        base_norm,
    ])

    # ==========================================================
    # OTHER DATASET TRAINING TRANSFORMS
    # ==========================================================
    transform_xray = transforms.Compose([
        transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.RandomApply([transforms.RandomAffine(degrees=8, translate=(0.05,0.05), scale=(0.92,1.08))], p=0.7),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        transforms.RandomResizedCrop(target_size, scale=(0.85,1.0), ratio=(0.9,1.1), antialias=True),
        transforms.ToTensor(),
        base_norm,
    ])

    transform_mri_roi = transforms.Compose([
        transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.RandomApply(
            [transforms.RandomRotation(degrees=6)],
            p=0.60,
        ),
        transforms.RandomApply(
            [
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.04, 0.04),
                    scale=(0.95, 1.05),
                )
            ],
            p=0.60,
        ),
        transforms.ToTensor(),
        base_norm,
    ])

    # ==========================================================
    # TRAIN TRANSFORM DICTIONARY
    # ==========================================================
    transform_dict = {
        "XRay": transform_xray,
        "MRI": transform_standard,
        "LSS_MRI_ROI_CLS": transform_mri_roi,

        # Keep BUS-BRA deterministic here.
        # It will normally not be called because __getitem__()
        # uses BUSBRAJointSegTransform for BUS-BRA segmentation and masked classification samples.
        "BUS_BRA": transform_standard,

        "default": transform_standard,
    }

    # ==========================================================
    # VALIDATION / TEST TRANSFORM DICTIONARY
    # ==========================================================
    # No random augmentation during validation or testing.
    val_transform_dict = {
        "XRay": transform_standard,
        "MRI": transform_standard,
        "LSS_MRI_ROI_CLS": transform_standard,
        "BUS_BRA": transform_standard,
        "default": transform_standard,
    }
    
    # ------------------------------------------------------------
    # Dataset Setup
    # ------------------------------------------------------------
    train_ds = UnifiedMedicalDataset(data_root, split='train', transform_dict=transform_dict, tokenizer=tokenizer, target_size=target_size, cache_images_in_ram=False, cache_targets_in_ram=True, use_report_label_guidance=USE_REPORT_LABEL_GUIDANCE)
    val_ds = UnifiedMedicalDataset(data_root, split='val', transform_dict=val_transform_dict, tokenizer=tokenizer, target_size=target_size, cache_images_in_ram=True, cache_targets_in_ram=True, use_report_label_guidance=USE_REPORT_LABEL_GUIDANCE)
    test_ds = UnifiedMedicalDataset(data_root, split='test', transform_dict=val_transform_dict, tokenizer=tokenizer, target_size=target_size, cache_images_in_ram=True, cache_targets_in_ram=True, use_report_label_guidance=USE_REPORT_LABEL_GUIDANCE)

    _dataset_fingerprints = {
        "train": dataset_membership_fingerprint(train_ds),
        "val": dataset_membership_fingerprint(val_ds),
        "test": dataset_membership_fingerprint(test_ds),
    }
    print(f"[Reproducibility] dataset_fingerprints={_dataset_fingerprints}")

    #debug detect
    # debug_detection_dataset(train_ds, num_samples=5)
    
    # Extract Class Names
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    if len(train_ds) == 0:
        print("Error: Dataset empty."); return
        
    num_modalities = len(train_ds.modality_map)
    
    pos_weight = compute_pos_weight_v3(
        train_ds,
        min_weight=1.0,       # was 0.3 — never down-weight a positive class
        max_weight=10.0,      # was 5.0 — rare classes need more signal
        smoothing="sqrt",     # sqrt: 100:1 ratio -> 10:1 (better than log's 4.6:1)
        smooth_scale=2.0,
        problem_weight=0.7,   # was 0.6
        force_problem_to_min=False,
    ).to(device, non_blocking=True)
    print("[CLS] pos_weight stats:",
        f"min={pos_weight.min().item():.2f}, max={pos_weight.max().item():.2f}, mean={pos_weight.mean().item():.2f}")
    
    roi_cls_weights = compute_local_multiclass_weights(
        train_ds,
        ds_name="LSS_MRI_ROI_CLS",
        num_local_classes=3,
        sqrt=True,
    ).to(device, non_blocking=True)

    print("MRI ROI crop class weights:", roi_cls_weights.tolist())

    _train_generator = torch.Generator()
    _train_generator.manual_seed(int(experiment_seed))

    batch_sampler = EqualModalityBatchSampler(
        dataset=train_ds,
        batch_size=unified_batch_size,
        modality_batch_counts=modality_batch_counts,
        batches_per_epoch=batches_per_epoch,
        shuffle=True,
        drop_last=True,
        random_seed=int(experiment_seed),
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        collate_fn=unified_collate,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        generator=_train_generator,
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=unified_collate)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=unified_collate) if len(test_ds)>0 else None

    seg_num_classes = len(train_ds.seg_class_names)
    print(f"Segmentation Classes ({seg_num_classes}): {train_ds.seg_class_names}")
    # mri_roi_class_weights = estimate_mri_severity_class_weights(train_loader, num_classes=4).to(device, non_blocking=True)
    # print("MRI ROI class weights:", mri_roi_class_weights.tolist())
    
    # ------------------------------------------------------------
    # Model Initialization
    # ------------------------------------------------------------
    
    # 1. Initialize Swin Transformer
    swin = SwinTransformer(
        img_size=target_size[0], patch_size=4, in_chans=3, 
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7
    )
    
    seg_num_classes = len(train_ds.seg_class_names)
    num_classes = len(train_ds.class_names)
    num_modalities = len(train_ds.modality_map)
    det_num_classes = len(train_ds.det_class_names)

    dino_repo_path, dino_cfg_path, dino_pretrain_path
    # 2. Initialize Model
    # Note: We pass medclip_path here so SwinMultiTask loads it using the NEW 'Rosetta Stone' loader

    # ---- BUS-BRA loss weighting aligned with the actual training sampler ----
    # The fixed-composition sampler already provides dedicated balanced
    # BI-RADS and pathology slots. Therefore weights are estimated from the
    # complete sampled epoch (all three BUS-BRA slots), not from raw dataset
    # frequencies. This corrects residual imbalance without double-weighting.
    _busbra_stream = estimate_busbra_training_stream_counts(
        train_ds,
        batch_sampler=batch_sampler,
        num_birads_classes=getattr(train_ds, 'birads_num_classes', 4),
    )

    if busbra_use_class_weights:
        _birads_class_weights_cpu = compute_birads_class_weights(
            train_ds,
            num_classes=getattr(train_ds, 'birads_num_classes', 4),
            counts=_busbra_stream['sampled_birads'],
            power=birads_weight_power,
            min_weight=busbra_class_weight_min,
            max_weight=busbra_class_weight_max,
        )
        _pathology_class_weights_cpu = compute_pathology_class_weights(
            train_ds,
            num_classes=2,
            counts=_busbra_stream['sampled_pathology'],
            power=pathology_weight_power,
            malignant_boost=pathology_malignant_boost,
            min_weight=busbra_class_weight_min,
            max_weight=busbra_class_weight_max,
        )
    else:
        _birads_class_weights_cpu = torch.ones(
            getattr(train_ds, 'birads_num_classes', 4), dtype=torch.float32
        )
        _pathology_class_weights_cpu = torch.ones(2, dtype=torch.float32)
        print('[BUS-BRA weighting] disabled; using uniform loss weights.')

    _birads_class_weights = _birads_class_weights_cpu.to(device)
    _pathology_class_weights = _pathology_class_weights_cpu.to(device)

    # Initialize both heads to the same effective distribution targeted by
    # the sampled and weighted losses.
    _birads_bias_init = compute_birads_bias_init(
        train_ds,
        num_classes=getattr(train_ds, 'birads_num_classes', 4),
        counts=_busbra_stream['sampled_birads'],
        class_weights=_birads_class_weights_cpu,
    )
    _pathology_bias_init = compute_pathology_bias_init(
        _busbra_stream['sampled_pathology'],
        class_weights=_pathology_class_weights_cpu,
    )

    _ultrasound_modality_id = None
    for _us_name in ("Ultrasound", "ultrasound", "US", "us"):
        if _us_name in train_ds.modality_map:
            _ultrasound_modality_id = int(train_ds.modality_map[_us_name])
            break
    print(f"[Ultrasound] modality_id={_ultrasound_modality_id}")

    model = SwinMultiTask(
        swin_model=swin,
        medclip_path=medclip_weights_path,
        use_moe=use_moe,
        moe_type=moe_type,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        gradif_attn_dim=gradif_attn_dim,
        gradif_num_heads=gradif_num_heads,
        gradif_router_dim=gradif_router_dim,
        gradif_top_k=gradif_top_k,
        gradif_routing_mode=gradif_routing_mode,
        gradif_expert_bank_type=gradif_expert_bank_type,
        gradif_temperature=gradif_temperature,
        gradif_modality_scale=gradif_modality_scale,
        gradif_router_strategy=gradif_router_strategy,
        gradif_router_task_residual_scale=(
            gradif_router_task_residual_scale
        ),
        gradif_router_modality_residual_scale=(
            gradif_router_modality_residual_scale
        ),
        gradif_router_attention_pool=gradif_router_attention_pool,
        gradif_router_alpha_content_init=gradif_router_alpha_content_init,
        gradif_router_alpha_task_init=gradif_router_alpha_task_init,
        gradif_router_alpha_modality_init=gradif_router_alpha_modality_init,
        gradif_router_task_specific_gates=gradif_router_task_specific_gates,
        gradif_router_modality_specific_gates=gradif_router_modality_specific_gates,
        gradif_router_modality_content_adapters=gradif_router_modality_content_adapters,
        gradif_router_branch_hidden=gradif_router_branch_hidden,
        mofx_router_configuration=mofx_router_configuration,
        gradif_router_mi_min_group_size=gradif_router_mi_min_group_size,
        gradif_router_mi_queue_size=gradif_router_mi_queue_size,
        gradif_router_mi_current_fraction=gradif_router_mi_current_fraction,
        gradif_gamma_init=0.015,
        gradif_gamma_min=0.0,
        gradif_gamma_max=0.12,
        gradif_expert_scale_init=0.35,
        gradif_expert_scale_max=1.0,
        gradif_generic_mlp_bottleneck=gradif_generic_mlp_bottleneck,
        gradif_include_generalist=gradif_include_generalist,
        gradif_include_null_expert=gradif_include_null_expert,
        gradif_use_granularity_prior=gradif_use_granularity_prior,
        gradif_decomposition=gradif_decomposition,
        gradif_active_stages=tuple(gradif_active_stages),
        gradif_fft_initial_boundaries=gradif_fft_initial_boundaries,
        gradif_fft_mask_temperature=gradif_fft_mask_temperature,
        gradif_fft_min_bandwidth=gradif_fft_min_bandwidth,
        num_modalities=num_modalities,
        ultrasound_modality_id=_ultrasound_modality_id,
        ultrasound_adapter_alpha_init=0.10,
        class_names=train_ds.class_names,
        seg_num_classes=seg_num_classes,
        seg_img_size=target_size[0],
        det_num_classes=det_num_classes,
        # det_num_queries=100,
        dino_repo_path=dino_repo_path,
        dino_cfg_path=dino_cfg_path,
        dino_pretrained_path=dino_pretrain_path,
        use_official_dino=True,
        use_cross_roi_reasoning=USE_CROSS_ROI_REASONING,
        llm_model_name=llm_model_name,
        birads_bias_init=_birads_bias_init,
        pathology_bias_init=_pathology_bias_init,
        shared_head_seed=int(experiment_seed) + 1000,
        gradif_seg_safe_branch=True,
        segmentation_gradif_fusion_init=0.85,
        segmentation_gradif_max_delta_ratios=(0.05, 0.04, 0.05),
        classification_gradif_fusion_init=classification_gradif_fusion_init,
        classification_gradif_max_delta_ratio=classification_gradif_max_delta_ratio,
    ).to(device, non_blocking=True)

    _parameter_counts = parameter_count_summary(model)
    print(f"[Model] parameter_counts={_parameter_counts}")

    # Restore the experiment RNG after architecture-specific module creation
    # and optional common-state loading.  Data order and subsequent stochastic
    # operations therefore start from the same experiment seed in every run.
    set_reproducibility(seed=experiment_seed, deterministic=deterministic_run)

    if use_moe and str(moe_type).lower() == "gradif":
        _bank = model.task_moe.banks["c3"]
        print("[MoFX-Dense] Final architecture:")
        print(f"  expert_bank_type: {getattr(model.task_moe, 'expert_bank_type', gradif_expert_bank_type)}")
        print(f"  routing_mode: {getattr(model.task_moe, 'routing_mode', gradif_routing_mode)}")
        print(f"  decomposition: {_bank.decomposition_name}")
        print(f"  experts: {_bank.expert_names}")
        print(f"  functional experts: {len([e for e in _bank.expert_names if e not in {'null', 'generalist'}])}")
        print(f"  generalist enabled: {bool(gradif_include_generalist)}")
        print(f"  null enabled: {bool(gradif_include_null_expert)}")
        print(f"  active stages: {list(model.task_moe.active_stages)}")
        print(f"  release run: {str(run_study_name)}")
        print("  shared identity residual provides the common/general path")
    model.use_gt_descriptors = False
    # Per-run results artifact consumed by paper_figures.py
    RESULTS = {
        "run_name": run_name,
        "moe_type": str(moe_type).lower() if use_moe else "none",
        "use_moe": bool(use_moe),
        "experiment_seed": int(experiment_seed),
        "deterministic_run": bool(deterministic_run),
        "experiment_group_id": str(experiment_group_id),
        "run_study_name": str(run_study_name),
        "run_base_dir": os.path.abspath(str(run_base_dir)),
        "parameter_counts": _parameter_counts,
        "dataset_fingerprints": _dataset_fingerprints,
        "config": {
            "epochs": int(epochs),
            "unified_batch_size": int(unified_batch_size),
            "batches_per_epoch": int(len(train_loader)),
            "requested_batches_per_epoch": int(batches_per_epoch),
            "accum_steps": int(ACCUM_STEPS),
            "moe_num_experts": int(moe_num_experts),
            "moe_top_k": int(moe_top_k),
            "gradif_top_k": int(gradif_top_k),
            "gradif_routing_mode": str(gradif_routing_mode),
            "gradif_expert_bank_type": str(gradif_expert_bank_type),
            "gradif_num_experts": int(
                model.task_moe.banks["c3"].num_experts
            ) if use_moe and str(moe_type).lower() == "gradif" else None,
            "gradif_active_experts_per_sample": (
                int(model.task_moe.banks["c3"].num_experts)
                if use_moe and str(moe_type).lower() == "gradif"
                and gradif_routing_mode == "dense_soft"
                else int(gradif_top_k)
            ),
            "gradif_include_generalist": bool(gradif_include_generalist),
            "gradif_include_null_expert": bool(gradif_include_null_expert),
            "gradif_expert_warmup_epochs": int(gradif_expert_warmup_epochs),
            "gradif_temperature_start": float(gradif_temperature),
            "gradif_temperature_end": float(gradif_temperature_end),
            "gradif_router_strategy": str(gradif_router_strategy),
            "mofx_router_configuration": str(mofx_router_configuration),
            "gradif_router_alpha_content_init": float(gradif_router_alpha_content_init),
            "gradif_router_alpha_task_init": float(gradif_router_alpha_task_init),
            "gradif_router_alpha_modality_init": float(gradif_router_alpha_modality_init),
            "gradif_router_task_specific_gates": bool(gradif_router_task_specific_gates),
            "gradif_router_modality_specific_gates": bool(gradif_router_modality_specific_gates),
            "gradif_router_modality_content_adapters": bool(gradif_router_modality_content_adapters),
            "gradif_router_branch_hidden": int(gradif_router_branch_hidden),
            "gradif_router_task_residual_scale": float(
                gradif_router_task_residual_scale
            ),
            "gradif_router_modality_residual_scale": float(
                gradif_router_modality_residual_scale
            ),
            "gradif_router_attention_pool": bool(
                gradif_router_attention_pool
            ),
            "gradif_router_mi_min_group_size": int(
                gradif_router_mi_min_group_size
            ),
            "gradif_router_mi_queue_size": int(gradif_router_mi_queue_size),
            "gradif_router_mi_current_fraction": float(
                gradif_router_mi_current_fraction
            ),
            "gradif_routing_mi_enabled": bool(gradif_routing_mi_enabled),
            "gradif_routing_mi_weight_max": float(
                gradif_routing_mi_weight_max
            ),
            "gradif_routing_mi_start_epoch": int(
                gradif_routing_mi_start_epoch
            ),
            "gradif_routing_mi_ramp_end_epoch": int(
                gradif_routing_mi_ramp_end_epoch
            ),
            "gradif_routing_mi_decay_start_epoch": int(
                gradif_routing_mi_decay_start_epoch
            ),
            "gradif_routing_mi_end_epoch": int(
                gradif_routing_mi_end_epoch
            ),
            "gradif_generic_mlp_bottleneck": int(gradif_generic_mlp_bottleneck),
            "gradif_decomposition": str(gradif_decomposition).lower(),
            "gradif_active_stages": list(gradif_active_stages),
            "gradif_fft_initial_boundaries": [
                float(v) for v in gradif_fft_initial_boundaries
            ],
            "gradif_fft_mask_temperature": float(gradif_fft_mask_temperature),
            "gradif_fft_min_bandwidth": float(gradif_fft_min_bandwidth),
            "classification_gradif_fusion_init": float(classification_gradif_fusion_init),
            "classification_gradif_max_delta_ratio": float(classification_gradif_max_delta_ratio),
            "gradif_classification_no_harm_weight_max": float(gradif_classification_no_harm_weight),
            "gradif_segmentation_no_harm_weight_max": float(gradif_segmentation_no_harm_weight),
            "gradif_no_harm_start_epoch": int(gradif_no_harm_start_epoch),
            "gradif_no_harm_ramp_epochs": int(gradif_no_harm_ramp_epochs),
            "llm_model_name": str(llm_model_name),
        },
        "contexts": {},
        "val_history": [],
    }
    RESULTS["contexts"]["busbra_class_weighting"] = {
        "enabled": bool(busbra_use_class_weights),
        "weight_source": str(_busbra_stream["source"]),
        "raw_birads_counts": _busbra_stream["raw_birads"].astype(int).tolist(),
        "sampled_birads_counts": _busbra_stream["sampled_birads"].astype(int).tolist(),
        "birads_class_weights": _birads_class_weights_cpu.tolist(),
        "raw_pathology_counts": _busbra_stream["raw_pathology"].astype(int).tolist(),
        "sampled_pathology_counts": _busbra_stream["sampled_pathology"].astype(int).tolist(),
        "pathology_class_weights": _pathology_class_weights_cpu.tolist(),
        "birads_weight_power": float(birads_weight_power),
        "pathology_weight_power": float(pathology_weight_power),
        "pathology_malignant_boost": float(pathology_malignant_boost),
        "weight_clip": [float(busbra_class_weight_min), float(busbra_class_weight_max)],
    }
    _results_path = os.path.join(manager.run_dir, f"results_{run_name}.json")

    # BUS-BRA: set measurement normalization from the TRAIN set (val/test reuse these stats)
    try:
        _mm_stats, _ss_stats = train_ds.compute_measurement_stats()
        if _mm_stats is not None and hasattr(model, 'measurement_encoder'):
            model.measurement_encoder.set_stats(_mm_stats, _ss_stats)
            print('[BUS-BRA] measurement encoder normalization set from train set')
    except Exception as _e_stats:
        print('[BUS-BRA] could not set measurement stats:', _e_stats)
    
    # (Optional) enable fast math
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = not bool(deterministic_run)
    torch.backends.cudnn.deterministic = bool(deterministic_run)

    # Compile ONCE
    model.swin = torch.compile(model.swin, mode="reduce-overhead")
    # model.swin = torch.compile(model.swin, backend="eager")
    # model.seg_decoder = torch.compile(model.seg_decoder, backend="eager")
    # model.detector = torch.compile(model.detector, backend="eager")
    # model.mri_cls_neck = torch.compile(model.mri_cls_neck, backend="eager")
    # model.mri_cls_head = torch.compile(model.mri_cls_head, backend="eager")
    # model.query_to_roi = torch.compile(model.query_to_roi, backend="eager")
    # model.mri_cls_head_fused = torch.compile(model.mri_cls_head_fused, backend="eager")
    # model.mri_roi_to_report = torch.compile(model.mri_roi_to_report, backend="eager")
    # model.mri_global_to_report = torch.compile(model.mri_global_to_report, backend="eager")
    # -----------------------------------------------------------
    # Paper setup: fine-tune the encoder and task modules from the first epoch.
    # The base LLM remains frozen while its LoRA parameters are trainable.
    # -----------------------------------------------------------
    
    print("--- PAPER TRAINING SETUP ---")
    for param in model.swin.parameters():
        param.requires_grad = True

    for name, param in model.report_decoder.decoder.named_parameters():
        param.requires_grad = "lora" in name
        if param.requires_grad:
            param.data = param.data.float()

    for name, param in model.named_parameters():
        if "swin" in name or "report_decoder.decoder" in name:
            continue
        param.requires_grad = True
        param.data = param.data.float()

    for name, param in model.named_parameters():
        if any(k in name for k in ["task_moe", "visual_adapter", "cls_head", "seg_decoder", "lora"]):
            param.data = param.data.float()

    optimizer = create_optimizer(
        model,
        lr_backbone=paper_lrs["backbone"],
        lr_moe=paper_lrs["mofx"],
        lr_moe_gates=paper_lrs["mofx"],
        lr_seg=paper_lrs["segmentation"],
        lr_cls=paper_lrs["classification"],
        lr_report=paper_lrs["report"],
        lr_detector=paper_lrs["detector"],
        lr_rest=paper_lrs["report"],
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        threshold=float(scheduler_threshold),
    )
    use_amp = bool(mixed_precision and device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fns = {'seg_bce': nn.BCEWithLogitsLoss(), 'seg_dice': DiceLoss(), 'cls_focal': AdaptiveFocalLoss(alpha=0.5, gamma=2.0, label_smoothing=0.05)}
           
    # --- RESUME LOAD LOGIC ---
    start_epoch = 0
    if resume:
        resume_path = os.path.join(manager.ckpt_dir, "last_model.pth")
        if os.path.isfile(resume_path):
            print(f"loading checkpoint '{resume_path}'")
            # Load to CPU first to avoid GPU OOM during loading
            checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                print("  [Info] Detected Rich Checkpoint.")
                start_epoch = checkpoint.get('epoch', 0) + 1
                
                # Use strict=False to allow loading if you made minor architecture tweaks
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("  [OK] Model weights loaded.")
                
                if 'optimizer_state_dict' in checkpoint:
                    saved_groups = len(checkpoint['optimizer_state_dict']['param_groups'])
                    current_groups = len(optimizer.param_groups)
                    print(f"  [Check] Optimizer Groups - Saved: {saved_groups}, Current: {current_groups}")
                    
                    try:
                        # Attempt to load optimizer state
                        # If sizes don't match or NaN poisoning is suspected, this will catch it
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("  [OK] Optimizer state loaded successfully.")
                    except Exception as e:
                        print(f"  [Warning] Optimizer resume failed: {e}")
                        print("  [Action] Proceeding with fresh Optimizer to ensure numerical stability.")
                        # We do NOT exit here; we just train with a fresh optimizer using loaded weights.
    
                if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except:
                        print("  [Info] Scheduler reset.")
                        
                if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    
                if 'best_val_loss' in checkpoint:
                    manager.best_val_loss = checkpoint['best_val_loss']
                if 'best_ultrasound_score' in checkpoint:
                    manager.best_ultrasound_score = float(
                        checkpoint['best_ultrasound_score']
                    )
                    
                print(f"  [+] Successfully Resumed from Epoch {start_epoch}")
            else:
                print("  [Warning] Detected Legacy Checkpoint. Loading weights only.")
                model.load_state_dict(checkpoint, strict=False)
                start_epoch = 0 

            del checkpoint
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f"  [!] No checkpoint found at '{resume_path}'. Starting from scratch.")


    if hasattr(train_ds, 'modality_map'):
        id_to_name_map = {v: k for k, v in train_ds.modality_map.items()}
    else:
        id_to_name_map = {0: 'XRay', 1: 'MRI', 2: 'CT', 3: 'US'} # Fallback
    print_teacher_map(
        num_modalities=4, 
        num_tasks=4, 
        class_names_map={0: 'XRay', 1: 'CT', 2: 'MRI', 3: 'Ultrasound'}
    )
    UNFREEZE_BACKBONE_EPOCH = 0

    # Report subsampling is intentionally not exposed in the final release.
    # EqualModalityBatchSampler guarantees report supervision every microbatch.
    print(f"[Report] LLM={llm_model_name} | training=every_microbatch")

    lora_unfrozen = any(("lora" in n) and p.requires_grad for n, p in model.report_decoder.named_parameters())
    backbone_unfrozen = any(p.requires_grad for p in model.swin.parameters())
    train_use_pred_boxes_for_mri_cls = False

    num_decoder_layers = getattr(model, "det_num_decoder_layers", 6)
    det_criterion = build_detection_criterion(
        det_num_classes=len(train_ds.det_class_names),
        device=device,
        num_decoder_layers=num_decoder_layers,
    )
    # -------------------------------
    # Training Loop
    # -------------------------------
    _optimizer_step_count = 0
    _skipped_bad_grad_steps = 0
    _empty_gradient_windows = 0
    print("\n=== Training Started ===")
    for epoch in range(start_epoch, epochs):
        _current_gradif_cls_no_harm_weight = 0.0
        _current_gradif_seg_no_harm_weight = 0.0
        _current_gradif_routing_mi_weight = 0.0
        if model.use_moe and getattr(model, "moe_type", None) == "gradif":
            _expert_warmup = epoch < max(0, int(gradif_expert_warmup_epochs))
            if hasattr(model.task_moe, "set_expert_warmup"):
                model.task_moe.set_expert_warmup(_expert_warmup, epoch=epoch)

            hold_epochs = max(
                0,
                int(gradif_temperature_hold_epochs),
                int(gradif_expert_warmup_epochs),
            )
            anneal_end = max(hold_epochs + 1, int(gradif_temperature_anneal_epochs))
            if epoch < hold_epochs:
                anneal_fraction = 0.0
            else:
                anneal_fraction = min(
                    max(float(epoch - hold_epochs) / float(anneal_end - hold_epochs), 0.0),
                    1.0,
                )
            current_gradif_temperature = (
                float(gradif_temperature)
                + anneal_fraction
                * (float(gradif_temperature_end) - float(gradif_temperature))
            )
            model.task_moe.set_temperature(current_gradif_temperature)
            _nh_start = max(
                int(gradif_no_harm_start_epoch),
                int(gradif_expert_warmup_epochs),
            )
            _nh_ramp = max(1, int(gradif_no_harm_ramp_epochs))
            _nh_progress = min(max((epoch - _nh_start + 1) / float(_nh_ramp), 0.0), 1.0)
            _current_gradif_cls_no_harm_weight = (
                float(gradif_classification_no_harm_weight) * _nh_progress
            )
            _current_gradif_seg_no_harm_weight = (
                float(gradif_segmentation_no_harm_weight) * _nh_progress
            )

            if bool(gradif_routing_mi_enabled):
                _mi_start = int(gradif_routing_mi_start_epoch)
                _mi_ramp_end = max(
                    _mi_start, int(gradif_routing_mi_ramp_end_epoch)
                )
                _mi_decay_start = max(
                    _mi_ramp_end, int(gradif_routing_mi_decay_start_epoch)
                )
                _mi_end = max(
                    _mi_decay_start + 1, int(gradif_routing_mi_end_epoch)
                )
                _mi_max = max(0.0, float(gradif_routing_mi_weight_max))
                if epoch < _mi_start or epoch >= _mi_end:
                    _current_gradif_routing_mi_weight = 0.0
                elif epoch <= _mi_ramp_end:
                    _mi_progress = (epoch - _mi_start + 1) / float(
                        _mi_ramp_end - _mi_start + 1
                    )
                    _current_gradif_routing_mi_weight = _mi_max * _mi_progress
                elif epoch < _mi_decay_start:
                    _current_gradif_routing_mi_weight = _mi_max
                else:
                    _mi_remaining = (_mi_end - epoch) / float(
                        _mi_end - _mi_decay_start
                    )
                    _current_gradif_routing_mi_weight = (
                        _mi_max * max(0.0, min(_mi_remaining, 1.0))
                    )

            print(
                f"[MoFX][epoch={epoch}] "
                f"routing_mode={gradif_routing_mode} "
                f"expert_warmup={_expert_warmup} "
                f"router_temperature={current_gradif_temperature:.4f} "
                f"cls_no_harm={_current_gradif_cls_no_harm_weight:.4f} "
                f"seg_no_harm={_current_gradif_seg_no_harm_weight:.4f} "
                f"context_mi_weight={_current_gradif_routing_mi_weight:.6f}"
            )

        # Transition BUS-BRA descriptor heads from GT morphology/ROI masks to
        # the same predicted descriptors used at validation and deployment.
        _desc_warm = max(0, int(descriptor_gt_warmup_epochs))
        _desc_trans = max(0, int(descriptor_gt_transition_epochs))
        if _desc_warm == 0 and _desc_trans == 0:
            _desc_gt_prob = 0.0
        elif epoch < _desc_warm:
            _desc_gt_prob = 1.0
        else:
            _desc_gt_prob = max(
                0.0,
                1.0 - float(epoch - _desc_warm) / float(max(1, _desc_trans)),
            )
        if hasattr(model, "set_descriptor_gt_probability"):
            model.set_descriptor_gt_probability(_desc_gt_prob)
        print(
            f"[BUS-BRA][epoch={epoch}] descriptor_gt_probability="
            f"{_desc_gt_prob:.3f}"
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_stats = {'loss': [], 'seg_loss': [], 'cls_loss': [], 'rep_loss': [], 'det_loss': []}
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(pbar):
            dev = move_multitask_batch_to_device(batch, device)
            # debug detect
            if debug and step < 3 and has_det.any():
                print("\n[DET BATCH CHECK]")
                for i in range(B):
                    if has_det[i]:
                        t = det_targets[i]
                        print(f"sample={i} path={batch['image_path'][i]}")
                        print("  tasks:", batch["tasks"][i])
                        print("  img shape:", tuple(imgs[i].shape))
                        print("  boxes abs shape:", None if t is None else tuple(t["boxes"].shape))
                        print("  boxes norm shape:", None if t is None else tuple(t["boxes_norm"].shape))
                        print("  labels:", None if t is None else t["labels"].tolist())
            
            imgs = dev["imgs"]
            mod_ids = dev["mod_ids"]
            captions = dev["captions"]
            report_labels = dev["report_labels"]
            task_mask = dev["task_mask"]
            has_seg = dev["has_seg"]
            has_cls = dev["has_cls"]
            has_rep = dev["has_rep"]
            has_det = dev["has_det"]
            det_targets = dev["det_targets"]
            gt_boxes_for_roi = dev["gt_boxes_for_roi"]
            is_lss_mri = dev["is_lss_mri"]

            B = imgs.shape[0]

            # Final-release invariant: every microbatch trains the report path.
            # The sampler supplies at least one labelled report, while -100
            # labels keep non-report rows out of the language-model loss.
            if not has_rep.any():
                raise RuntimeError(
                    "Report-training contract violated: this microbatch has no "
                    "report-supervised sample."
                )
            if captions is None or report_labels is None:
                raise RuntimeError(
                    "Report-training contract violated: tokenized reports are "
                    "missing from a report-supervised microbatch."
                )
            _valid_report_rows = (report_labels != -100).any(dim=1)
            if not bool((has_rep & _valid_report_rows).any().item()):
                raise RuntimeError(
                    "Report-training contract violated: no report sample has "
                    "a valid target token."
                )

            _captions_fwd = captions
            _report_labels_fwd = report_labels
            _task_mask_fwd = task_mask
                        
            _need_gradif_reference = bool(
                model.use_moe
                and getattr(model, "moe_type", None) == "gradif"
                and (
                    float(_current_gradif_seg_no_harm_weight) > 0.0
                    or float(_current_gradif_cls_no_harm_weight) > 0.0
                    or float(gradif_classification_base_aux_weight) > 0.0
                )
                and (step % max(1, int(gradif_no_harm_every)) == 0)
            )

            with torch.amp.autocast('cuda'):
                outputs = model(imgs, modality_ids=mod_ids, task_mask=_task_mask_fwd, boxes=gt_boxes_for_roi, det_targets=det_targets, captions=_captions_fwd, 
                                report_labels=_report_labels_fwd, use_pred_boxes_for_mri_cls=(train_use_pred_boxes_for_mri_cls and bool(getattr(model, "use_cross_roi_reasoning", True))), 
                                det_score_thresh = 0.1, is_lss_mri=is_lss_mri,
                                measurements=batch['measurements'].to(device, non_blocking=True),
                                measurements_present=batch['measurements_present'].to(device, non_blocking=True),
                                measurement_sizes=batch.get('measurement_sizes'),
                                busbra_seg_channel_indices=batch.get('busbra_seg_channel_indices'),
                                has_birads=batch['has_birads'].to(device, non_blocking=True),
                                has_pathology=batch['has_pathology'].to(device, non_blocking=True),
                                seg_mask_gt=(batch['seg_mask'].to(device, non_blocking=True) if 'seg_mask' in batch else None),
                                gt_pathology_idx=batch['pathology_index'].to(device, non_blocking=True),
                                compute_gradif_reference=_need_gradif_reference)
                last_c4_shared = outputs["feat_c4_shared"].detach()
                #debug detect
                if debug and step % 20 == 0 and outputs.get("detection_active") is not None:
                    det_dbg = outputs["detection_active"]
                    p = det_dbg["pred_logits"].detach().float().sigmoid()
                    b = det_dbg["pred_boxes"].detach().float()

                    print("\n[DINO FORWARD CHECK]")
                    print("active batch idx:", outputs["det_batch_idx"].tolist())
                    print("pred_logits shape:", tuple(det_dbg["pred_logits"].shape))
                    print("pred_boxes shape:", tuple(det_dbg["pred_boxes"].shape))
                    print("prob max:", p.max().item(), "prob mean:", p.mean().item())
                    print("box min:", b.min().item(), "box max:", b.max().item())            
    
                W_SEG, W_CLS, W_REP = 2, 1.0, 1.0
                W_DET = 1.0
                task_losses = {}
                _final_seg_unweighted = None
                _final_cls_unweighted = None
                _seg_diag_cache = None   # cross-dataset seg conflict diagnostic (added)
                # 1. Segmentation Loss
                pred_masks_small = outputs.get("pred_masks_small", None)
                seg_pred_batch_idx = outputs.get("seg_pred_batch_idx", None)

                if pred_masks_small is not None and seg_pred_batch_idx is not None and 'seg_mask' in batch and has_seg.any():
                    # Which entries inside the segmentation subset actually have GT masks?
                    seg_supervised_local = has_seg[seg_pred_batch_idx]   # [N_seg]
                    if seg_supervised_local.any():
                        seg_gt_batch_idx = seg_pred_batch_idx[seg_supervised_local]  # batch positions with GT masks

                        preds = F.interpolate(
                            pred_masks_small[seg_supervised_local],
                            size=target_size,
                            mode='bilinear',
                            align_corners=False,
                        )
                        targets = batch['seg_mask'].to(device, non_blocking=True).float()[seg_gt_batch_idx]
                        active_ch = batch['seg_active_ch'].to(device, non_blocking=True).float()[seg_gt_batch_idx]  # [N, C]
                        l_bce = masked_bce_with_logits(preds, targets, active_ch)
                        l_dice = loss_fns['seg_dice'](preds, targets, active_channels=active_ch)
                        _seg_diag_cache = {   # references only (added)
                            "preds": preds, "targets": targets, "active_ch": active_ch,
                            "ds_name": [batch['ds_name'][int(_i)] for _i in seg_gt_batch_idx.tolist()],
                        }
                        l_seg_total = l_dice + l_bce
                        _final_seg_unweighted = l_seg_total
                        task_losses['seg'] = W_SEG * l_seg_total
                        train_stats['seg_loss'].append(l_seg_total.item())

                        _base_small = outputs.get("pred_masks_base_small", None)
                        if (
                            _base_small is not None
                            and float(_current_gradif_seg_no_harm_weight) > 0.0
                        ):
                            _base_preds = F.interpolate(
                                _base_small[seg_supervised_local],
                                size=target_size,
                                mode='bilinear',
                                align_corners=False,
                            )
                            _fused_per = segmentation_loss_per_sample(
                                preds, targets, active_ch
                            )
                            _base_per = segmentation_loss_per_sample(
                                _base_preds, targets, active_ch
                            )
                            _seg_no_harm = F.relu(
                                _fused_per - _base_per.detach()
                            ).mean()
                            if torch.isfinite(_seg_no_harm):
                                task_losses['seg_gradif_no_harm'] = (
                                    float(_current_gradif_seg_no_harm_weight)
                                    * _seg_no_harm
                                )
                                train_stats.setdefault(
                                    'seg_gradif_no_harm_loss', []
                                ).append(float(_seg_no_harm.detach().cpu()))

                        _stage_gates = outputs.get(
                            "segmentation_gradif_stage_gates", None
                        )
                        if (
                            _stage_gates is not None
                            and float(gradif_spatial_gate_identity_weight) > 0.0
                        ):
                            task_losses['seg_gradif_gate_identity'] = (
                                float(gradif_spatial_gate_identity_weight)
                                * _stage_gates.float().mean()
                            )

                        # if torch.isnan(l_bce) or torch.isnan(l_dice):
                        #     print(f"DEBUG: NaN in Seg Loss! BCE: {l_bce}, Dice: {l_dice}")

                # 2. Classification Loss #change
                _cls_diag_cache = None   # cross-dataset conflict diagnostic (added)
                if outputs.get('class_logits') is not None and has_cls.any():
                    logits_all = outputs['class_logits'].float()
                    labels_all = batch['class_labels'].to(device, non_blocking=True).float()
                    mask_all = batch['class_mask'].to(device, non_blocking=True).float()
                    cls_problem_type = batch['cls_problem_type'].to(device, non_blocking=True)
                    class_index_all = batch['class_index'].to(device, non_blocking=True)
                    local_class_indices_all = batch['local_class_indices'].to(device, non_blocking=True)
                    _cls_diag_cache = {   # references only (added)
                        "logits_all": logits_all, "labels_all": labels_all, "mask_all": mask_all,
                        "cls_problem_type": cls_problem_type, "class_index_all": class_index_all,
                        "local_class_indices_all": local_class_indices_all,
                        "has_cls": has_cls, "ds_name": batch['ds_name'],
                    }

                    cls_terms = []

                    # ---------------- multilabel branch ----------------
                    ml_mask = has_cls & (cls_problem_type == 0)
                    if ml_mask.any():
                        logits_ml = logits_all[ml_mask]
                        targets_ml = labels_all[ml_mask]
                        mask_ml = mask_all[ml_mask]

                        per_label_loss = loss_fns['cls_focal'](logits_ml, targets_ml, pos_weight=pos_weight)
                        masked_loss = per_label_loss * mask_ml
                        l_cls_ml = masked_loss.sum() / (mask_ml.sum() + 1e-8)

                        cls_terms.append(l_cls_ml)
                        train_stats.setdefault('cls_multilabel_loss', []).append(l_cls_ml.item())

                    # ---------------- multiclass branch ----------------
                    mc_mask = has_cls & (cls_problem_type == 1)
                    if mc_mask.any():
                        mc_losses = []

                        mc_indices = torch.where(mc_mask)[0].tolist()
                        for bi in mc_indices:
                            idxs = local_class_indices_all[bi]
                            idxs = idxs[idxs >= 0].long()

                            if idxs.numel() == 0:
                                continue

                            logits_i = logits_all[bi, idxs].unsqueeze(0)   # [1, K]
                            target_i = class_index_all[bi].view(1)         # [1]

                            if int(target_i.item()) < 0:
                                continue

                            if batch['ds_name'][bi] == "LSS_MRI_ROI_CLS":
                                loss_i = F.cross_entropy(logits_i, target_i, weight=roi_cls_weights)
                            else:
                                loss_i = F.cross_entropy(logits_i, target_i)

                            mc_losses.append(loss_i)

                        if len(mc_losses) > 0:
                            l_cls_mc = torch.stack(mc_losses).mean()
                            cls_terms.append(l_cls_mc)
                            train_stats.setdefault('cls_multiclass_loss', []).append(l_cls_mc.item())

                    if len(cls_terms) > 0:
                        l_cls_total = torch.stack(cls_terms).sum()
                        _final_cls_unweighted = l_cls_total
                        task_losses['cls'] = W_CLS * l_cls_total
                        train_stats['cls_loss'].append(l_cls_total.item())

                # 2b. MoFX numerical regularization. Expert selection is learned
                # only from downstream task losses; no routing-shape penalties
                # prescribe expert usage, task separation, or entropy.
                if model.use_moe and getattr(model, "moe_type", None) == "gradif":
                    gradif_regs = model.task_moe.regularization_losses()
                    warmup = max(
                        int(gradif_regularization_warmup),
                        int(gradif_expert_warmup_epochs),
                    )
                    ramp_den = max(1, int(gradif_temperature_anneal_epochs) - warmup)
                    reg_progress = min(
                        max((float(epoch) - warmup) / ramp_den, 0.0),
                        1.0,
                    )

                    if gradif_sparsity_weight > 0 and reg_progress > 0:
                        task_losses["gradif_sparsity"] = (
                            reg_progress
                            * gradif_sparsity_weight
                            * gradif_regs["sparsity"]
                        )
                    if gradif_load_balance_weight > 0:
                        task_losses["gradif_load_balance"] = (
                            gradif_load_balance_weight
                            * gradif_regs["load_balance"]
                        )
                    if gradif_task_separation_weight > 0:
                        task_losses["gradif_task_separation"] = (
                            gradif_task_separation_weight
                            * gradif_regs["task_separation"]
                        )
                    if gradif_expert_diversity_weight > 0:
                        task_losses["gradif_expert_diversity"] = (
                            gradif_expert_diversity_weight
                            * gradif_regs["expert_diversity"]
                        )
                    if gradif_gate_weight > 0:
                        task_losses["gradif_gate_usage"] = (
                            gradif_gate_weight * gradif_regs["gate_usage"]
                        )
                    # Kept for interface compatibility; configured to zero for
                    # naturally emerging input-conditioned specialization.
                    if (
                        gradif_router_conditioning_weight > 0
                        and "router_conditioning" in gradif_regs
                    ):
                        cond_scale = min(
                            max((float(epoch) - warmup) / 5.0, 0.0), 1.0
                        )
                        if cond_scale > 0:
                            task_losses["gradif_router_conditioning"] = (
                                cond_scale
                                * gradif_router_conditioning_weight
                                * gradif_regs["router_conditioning"]
                            )
                    # Temporary LIMoE-style conditional mutual-information
                    # objective. It is calculated independently inside each
                    # stage x route x modality context using clean probabilities.
                    # A detached rolling queue makes it active even when the
                    # fixed medical sampler has only 1--3 examples per context.
                    if (
                        _current_gradif_routing_mi_weight > 0.0
                        and "context_mi_loss" in gradif_regs
                    ):
                        task_losses["gradif_context_mi"] = (
                            _current_gradif_routing_mi_weight
                            * gradif_regs["context_mi_loss"]
                        )

                    # Direct L2 pull on router log-sum-exp for numerical stability.
                    # This does not prescribe expert usage.
                    # v28: direct L2 pull on the raw gate logit. Applied from
                    # step 1, no warmup ramp -- this isn't fighting for
                    # conditional variance, it's just keeping the router's
                    # pre-activation logit from running away unboundedly, so
                    # router_conditioning's own gradient stays usable.
                    if (
                        gradif_logit_magnitude_weight > 0
                        and "logit_magnitude" in gradif_regs
                    ):
                        task_losses["gradif_logit_magnitude"] = (
                            gradif_logit_magnitude_weight
                            * gradif_regs["logit_magnitude"]
                        )

                    _route_log_every = max(1, int(gradif_runtime_stats_every))
                    if step % _route_log_every == 0:
                        for reg_name, reg_value in gradif_regs.items():
                            train_stats.setdefault(
                                f"gradif_{reg_name}_loss", []
                            ).append(float(reg_value.detach().cpu()))
                        train_stats.setdefault("gradif_reg_progress", []).append(reg_progress)
                        train_stats.setdefault(
                            "gradif_context_mi_weight", []
                        ).append(float(_current_gradif_routing_mi_weight))

                        for _stage, _bank in model.task_moe.banks.items():
                            for _tid, _weights in _bank.latest_weights.items():
                                if _weights.numel() == 0:
                                    continue
                                _p = _weights.detach().float().clamp_min(1e-8)
                                _entropy = -(
                                    _p * _p.log()
                                ).sum(dim=-1).mean() / math.log(max(_p.shape[-1], 2))
                                train_stats.setdefault(
                                    f"gradif_{_stage}_task{_tid}_entropy", []
                                ).append(float(_entropy.cpu()))
                                _gate = _bank.latest_gates.get(_tid)
                                if _gate is not None and _gate.numel() > 0:
                                    train_stats.setdefault(
                                        f"gradif_{_stage}_task{_tid}_gate", []
                                    ).append(float(_gate.detach().float().mean().cpu()))

                # 3. Report loss is mandatory on every microbatch.
                l_rep = outputs.get('rep_loss')
                if l_rep is None:
                    raise RuntimeError(
                        "The report decoder did not return a loss for a "
                        "report-supervised microbatch."
                    )
                if not torch.isfinite(l_rep):
                    raise FloatingPointError(
                        "The report decoder returned a non-finite loss."
                    )
                task_losses['rep'] = W_REP * l_rep
                train_stats['rep_loss'].append(l_rep.item())
                
                # 3b. Dedicated BUS-BRA heads. BI-RADS and pathology have
                # independent supervision masks; neither depends on the generic
                # classifier and pathology no longer depends on has_birads.
                W_BIRADS, W_PATH = 1.0, 1.0
                if outputs.get('birads_logits') is not None and 'has_birads' in batch:
                    hb = batch['has_birads'].to(device, non_blocking=True)
                    if hb.any():
                        bl_ = outputs['birads_logits'].float()
                        bord_ = batch['birads_ordinal'].to(device, non_blocking=True).float()
                        _bgt_w_idx = batch['birads_index'].to(
                            device, non_blocking=True
                        )
                        _birads_sample_w = torch.ones(
                            _bgt_w_idx.shape[0], device=device, dtype=bl_.dtype
                        )
                        _valid_bw = hb & (_bgt_w_idx >= 0) & (
                            _bgt_w_idx < _birads_class_weights.numel()
                        )
                        if _valid_bw.any():
                            _birads_sample_w[_valid_bw] = _birads_class_weights[
                                _bgt_w_idx[_valid_bw]
                            ].to(dtype=bl_.dtype)
                        l_birads = birads_ordinal_loss(
                            bl_, bord_, mask=hb, sample_weight=_birads_sample_w
                        )
                        if torch.isfinite(l_birads):
                            task_losses['birads'] = W_BIRADS * l_birads
                            train_stats.setdefault('birads_loss', []).append(l_birads.item())

                        # Classification-safe GraDiF supervision.  The identity
                        # branch is trained as a stable baseline, and the fused
                        # GraDiF output is penalized whenever its per-sample
                        # ordinal loss exceeds that baseline.
                        _base_bl = outputs.get('birads_base_logits', None)
                        if (
                            getattr(model.module if hasattr(model, "module") else model,
                                    "moe_type", None) == "gradif"
                            and _base_bl is not None
                        ):
                            _fused_per = F.binary_cross_entropy_with_logits(
                                bl_, bord_, reduction="none"
                            ).sum(dim=-1)
                            _base_per = F.binary_cross_entropy_with_logits(
                                _base_bl.float(), bord_, reduction="none"
                            ).sum(dim=-1)
                            _safe_w = hb.float() * _birads_sample_w
                            _safe_den = _safe_w.sum().clamp_min(1.0)
                            _base_aux = (_base_per * _safe_w).sum() / _safe_den
                            _no_harm = (
                                F.relu(_fused_per - _base_per.detach()) * _safe_w
                            ).sum() / _safe_den
                            if (
                                float(gradif_classification_base_aux_weight) > 0.0
                                and torch.isfinite(_base_aux)
                            ):
                                task_losses['birads_base_aux'] = (
                                    float(gradif_classification_base_aux_weight)
                                    * _base_aux
                                )
                            if torch.isfinite(_no_harm):
                                task_losses['birads_no_harm'] = (
                                    float(_current_gradif_cls_no_harm_weight)
                                    * _no_harm
                                )
                            train_stats.setdefault('birads_base_aux_loss', []).append(
                                float(_base_aux.detach().cpu())
                            )
                            train_stats.setdefault('birads_no_harm_loss', []).append(
                                float(_no_harm.detach().cpu())
                            )

                        _core_model = model.module if hasattr(model, "module") else model
                        if hasattr(_core_model, "birads_head"):
                            BIRADS_BIAS_MARGIN = 0.15
                            W_BIRADS_SEP = 0.05
                            _biases = _core_model.birads_head.biases
                            if _biases.numel() >= 2:
                                _gap = _biases[:-1] - _biases[1:]
                                l_birads_sep = F.relu(
                                    BIRADS_BIAS_MARGIN - _gap
                                ).mean()
                                if torch.isfinite(l_birads_sep):
                                    task_losses['birads_bias_sep'] = (
                                        W_BIRADS_SEP * l_birads_sep
                                    )
                                    train_stats.setdefault(
                                        'birads_bias_sep_loss', []
                                    ).append(l_birads_sep.item())

                pml_ = outputs.get('pathology_meas_logits', None)
                if (
                    pml_ is not None
                    and 'has_pathology' in batch
                    and 'pathology_index' in batch
                ):
                    hp = batch['has_pathology'].to(device, non_blocking=True)
                    pidx_ = batch['pathology_index'].to(device, non_blocking=True)
                    valid_p = hp & (pidx_ >= 0)
                    if valid_p.any():
                        l_path = F.cross_entropy(
                            pml_[valid_p].float(), pidx_[valid_p],
                            weight=_pathology_class_weights,
                        )
                        if torch.isfinite(l_path):
                            task_losses['pathology'] = W_PATH * l_path
                            train_stats.setdefault('pathology_loss', []).append(
                                l_path.item()
                            )

                        _base_path = outputs.get('pathology_base_logits', None)
                        if (
                            getattr(model.module if hasattr(model, "module") else model,
                                    "moe_type", None) == "gradif"
                            and _base_path is not None
                        ):
                            _fused_path_per = F.cross_entropy(
                                pml_[valid_p].float(), pidx_[valid_p],
                                weight=_pathology_class_weights,
                                reduction='none',
                            )
                            _base_path_per = F.cross_entropy(
                                _base_path[valid_p].float(), pidx_[valid_p],
                                weight=_pathology_class_weights,
                                reduction='none',
                            )
                            _path_base_aux = _base_path_per.mean()
                            _path_no_harm = F.relu(
                                _fused_path_per - _base_path_per.detach()
                            ).mean()
                            if (
                                float(gradif_classification_base_aux_weight) > 0.0
                                and torch.isfinite(_path_base_aux)
                            ):
                                task_losses['pathology_base_aux'] = (
                                    float(gradif_classification_base_aux_weight)
                                    * _path_base_aux
                                )
                            if torch.isfinite(_path_no_harm):
                                task_losses['pathology_no_harm'] = (
                                    float(_current_gradif_cls_no_harm_weight)
                                    * _path_no_harm
                                )
                            train_stats.setdefault(
                                'pathology_base_aux_loss', []
                            ).append(float(_path_base_aux.detach().cpu()))
                            train_stats.setdefault(
                                'pathology_no_harm_loss', []
                            ).append(float(_path_no_harm.detach().cpu()))

                # 4. Detection Loss (MRI only, OFFICIAL DINO ONLY)
                det_match_indices = None
                active_to_batch_idx = None

                det_active_idx = outputs.get("det_batch_idx", None)

                if (
                    outputs.get("detection", None) is not None
                    and det_targets is not None
                    and det_active_idx is not None
                    and det_active_idx.numel() > 0
                ):
                    # Build the exact active mask used by forward()
                    det_active_mask = torch.zeros(imgs.shape[0], dtype=torch.bool, device=device)
                    det_active_mask[det_active_idx] = True
                    active_to_batch_idx = det_active_idx

                    det_targets_active, _ = prepare_detection_targets(
                        det_targets,
                        det_active_mask,
                        imgs.shape[-2],
                        imgs.shape[-1],
                        num_classes=det_num_classes,
                        device=device,
                    )

                    # outputs["detection"] is ACTIVE-ONLY during training
                    # pred_logits: [N_det_active, Q, C]
                    # pred_boxes : [N_det_active, Q, 4]
                    if len(det_targets_active) == det_active_idx.numel():
                        det_out_src = outputs["detection"]

                        det_out = {
                            "pred_logits": det_out_src["pred_logits"].float(),
                            "pred_boxes": det_out_src["pred_boxes"].float(),
                        }

                        if "aux_outputs" in det_out_src and det_out_src["aux_outputs"] is not None:
                            det_out["aux_outputs"] = [
                                {
                                    "pred_logits": aux["pred_logits"].float(),
                                    "pred_boxes": aux["pred_boxes"].float(),
                                }
                                for aux in det_out_src["aux_outputs"]
                            ]

                        if "dn_meta" in det_out_src:
                            det_out["dn_meta"] = det_out_src["dn_meta"]

                        with torch.cuda.amp.autocast(enabled=False):                                  
                            det_loss_dict, indices_list = model.detector.compute_loss(
                                det_out,
                                det_targets_active,
                                return_indices=True,
                            )

                        det_match_indices = (
                            indices_list[-1]
                            if isinstance(indices_list, list) and len(indices_list) > 0
                            else None
                        )

                        l_det = torch.zeros((), device=device)
                        for k, v in det_loss_dict.items():
                            if k in model.detector.criterion.weight_dict:
                                l_det = l_det + model.detector.criterion.weight_dict[k] * v

                        if torch.isfinite(l_det):
                            task_losses["det"] = W_DET * l_det
                            train_stats["det_loss"].append(float(l_det.item()))
                        else:
                            print("DEBUG: NaN/Inf in Detection Loss!")
                    else:
                        print(
                            f"DEBUG: detection target count mismatch: "
                            f"active_idx={det_active_idx.numel()} vs det_targets_active={len(det_targets_active)}"
                        )
                    #debug detect
                    if debug and det_match_indices is not None and step % 20 == 0:
                        print("\n[DINO MATCH/Loss CHECK]")
                        for local_i, (src_idx, tgt_idx) in enumerate(det_match_indices):
                            gt_n = det_targets_active[local_i]["labels"].numel()
                            print(f"local={local_i} batch={int(active_to_batch_idx[local_i])} matched={len(src_idx)} gt={gt_n}")

                        for k, v in det_loss_dict.items():
                            print(f"{k}: {float(v.detach().cpu())}")
                            
            # Standard joint multitask loss with gradient accumulation.
            num_steps = step + 1
            is_last_batch = num_steps == len(train_loader)
            should_step = ((num_steps % ACCUM_STEPS) == 0) or is_last_batch

            total_loss = torch.tensor(0.0, device=device)
            for name, task_loss in task_losses.items():
                if task_loss.requires_grad:
                    total_loss = total_loss + task_loss
                    train_stats.setdefault(f"{name}_loss", []).append(
                        float(task_loss.detach().cpu())
                    )

            nonfinite_loss_names = [
                name for name, task_loss in task_losses.items()
                if task_loss.requires_grad
                and not torch.isfinite(task_loss.detach()).all()
            ]
            if nonfinite_loss_names:
                print(
                    f"[Numerics][epoch={epoch} step={step}] non-finite losses: "
                    + ", ".join(nonfinite_loss_names)
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if total_loss.requires_grad:
                scaled_loss = total_loss / ACCUM_STEPS
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            # -----------------------------
            # Optim step (per ACCUM_STEPS)
            # -----------------------------
            if should_step:
                # Some accumulation windows may contain no differentiable losses.
                # In that case GradScaler has no inf/NaN check to consume.
                trainable = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]

                if len(trainable) == 0:
                    print(
                        f"[AMP][epoch={epoch} step={step}] "
                        "No gradients in this accumulation window; "
                        "skipping optimizer/scaler update."
                    )
                    _empty_gradient_windows += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Unscale once, then check finiteness BEFORE clipping. If a
                # single gradient contains Inf, global-norm clipping can use a
                # zero coefficient and turn Inf*0 into NaN across many tensors,
                # hiding the true source (often making patch_embed appear first).
                scaler.unscale_(optimizer)

                found_bad_grad = False
                bad_grad_name = None
                bad_grad_kind = None
                bad_grad_max_abs = None
                sanitized_control_grads = []

                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue

                    finite_mask = torch.isfinite(p.grad)
                    if not finite_mask.all():
                        is_control = (
                            p.numel() <= 16
                            or "fusion_logits" in name
                            or ".router." in name
                        )
                        if is_control:
                            p.grad.data = torch.nan_to_num(
                                p.grad.data.float(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            ).to(p.grad.dtype)
                            sanitized_control_grads.append(name)
                            continue
                        found_bad_grad = True
                        bad_grad_name = name
                        _g = p.grad.detach()
                        bad_grad_kind = (
                            "NaN" if torch.isnan(_g).any()
                            else "Inf"
                        )
                        _finite_values = _g[torch.isfinite(_g)]
                        bad_grad_max_abs = (
                            float(_finite_values.abs().max().cpu())
                            if _finite_values.numel() else float("nan")
                        )
                        break

                if sanitized_control_grads:
                    train_stats.setdefault(
                        "sanitized_control_gradient_count", []
                    ).append(float(len(sanitized_control_grads)))
                    print(
                        "[AMP] Sanitized non-finite control gradients: "
                        + ", ".join(sanitized_control_grads)
                    )

                if found_bad_grad:
                    print("\n[NONFINITE GRADIENT FOUND BEFORE CLIPPING]")
                    print("parameter:", bad_grad_name)
                    print("kind:", bad_grad_kind, "finite_max_abs:", bad_grad_max_abs)
                    print("amp_scale:", float(scaler.get_scale()) if scaler.is_enabled() else 1.0)
                    print("[SKIPPING OPTIMIZER STEP DUE TO BAD GRADIENTS]")

                    _skipped_bad_grad_steps += 1
                    optimizer.zero_grad(set_to_none=True)

                    if scaler is not None and scaler.is_enabled():
                        old_scale = float(scaler.get_scale())
                        scaler.update(new_scale=max(old_scale * 0.5, 1.0))

                    continue

                # Clip only verified-finite gradients.
                if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        max_norm=GRAD_CLIP_NORM,
                        error_if_nonfinite=True,
                    )
                    train_stats.setdefault("grad_norm", []).append(
                        float(grad_norm.detach().cpu())
                    )

                # Normal AMP optimizer step.
                scale_before = (
                    float(scaler.get_scale())
                    if scaler.is_enabled()
                    else 1.0
                )

                scaler.step(optimizer)
                scaler.update()
                _optimizer_step_count += 1

                scale_after = (
                    float(scaler.get_scale())
                    if scaler.is_enabled()
                    else scale_before
                )

                # If the scale decreased, GradScaler skipped optimizer.step()
                # because it detected an overflow.
                optimizer_step_skipped = (
                    scaler.is_enabled()
                    and scale_after < scale_before
                )

                if (
                    not optimizer_step_skipped
                    and model.use_moe
                    and getattr(model, "moe_type", None) in {"sparse", "gradif"}
                    and model.task_moe is not None
                    and hasattr(model.task_moe, "step_bias")
                ):
                    model.task_moe.step_bias()

                optimizer.zero_grad(set_to_none=True)

            # -----------------------------
            # Logging
            # -----------------------------
            total_loss_val = float(total_loss.detach().cpu().item()) if total_loss.requires_grad else 0.0
            train_stats['loss'].append(total_loss_val)
            if step % 20 == 0:
                pbar.set_postfix({
                    'Tot': f"{total_loss_val:.3f}",
                    'Seg': f"{(task_losses.get('seg', torch.tensor(0., device=device))).detach().cpu().item():.3f}",
                    'Cls': f"{(task_losses.get('cls', torch.tensor(0., device=device))).detach().cpu().item():.3f}",
                    'Rep': f"{(task_losses.get('rep', torch.tensor(0., device=device))).detach().cpu().item():.3f}",
                    'Det': f"{(task_losses.get('det', torch.tensor(0., device=device))).detach().cpu().item():.3f}",
                })
            # print(f"\n[Epoch {epoch}] mean_train_loss={np.mean(train_stats['loss']):.4f}")
                        
            # --- DEBUG VISUALIZATION ---
            # if epoch < 2 and step < 5:
            #     print(f"\n[Batch Debug] Epoch={epoch} Step={step}")
            #     for bi in range(len(batch['tasks'])):
            #         print(
            #             f"  sample={bi} | ds={batch['ds_name'][bi]} | "
            #             f"modality_id={batch['modality_id'][bi].item()} | tasks={batch['tasks'][bi]}"
                    # )
            # if epoch < 3 and step < 5:
            #     print(f"\n[Debug][Epoch {epoch}][Step {step}] Batch tasks:")
            #     for bi in range(len(batch['tasks'])):
            #         print(f"  sample {bi}: ds={batch['ds_name'][bi]} | tasks={batch['tasks'][bi]}")
                
            if debug and (step % 10 == 0) and ('detection' in outputs) and (det_targets is not None):
                try:
                    core_model = model.module if hasattr(model, "module") else model
                    has_det_cpu = ['detection' in str(t).lower() for t in batch['tasks']]
                    img_h, img_w = imgs.shape[-2], imgs.shape[-1]

                    for b_idx in range(len(has_det_cpu)):
                        if not has_det_cpu[b_idx]:
                            continue

                        gt_boxes_b = det_targets[b_idx]['boxes'].detach()
                        gt_labels_b = det_targets[b_idx]['labels'].detach()

                        # Prefer already-decoded predictions if forward created them
                        if (
                            outputs.get("mri_pred_boxes", None) is not None and
                            outputs.get("mri_pred_labels", None) is not None and
                            outputs.get("mri_pred_scores", None) is not None
                        ):
                            pred_boxes_abs = outputs["mri_pred_boxes"][b_idx].detach()
                            pred_labels_b = outputs["mri_pred_labels"][b_idx].detach()
                            pred_scores_b = outputs["mri_pred_scores"][b_idx].detach()
                            pred_query_ids_b = outputs.get("mri_pred_query_ids", [None] * len(has_det_cpu))[b_idx]
                        else:
                            pred_logits_b = outputs['detection']['pred_logits'][b_idx].detach().float()
                            pred_boxes_b  = outputs['detection']['pred_boxes'][b_idx].detach().float()

                            pred_boxes_abs, pred_labels_b, pred_scores_b, pred_query_ids_b = core_model.postprocess_dino_predictions(
                                pred_logits_b,
                                pred_boxes_b,
                                img_w,
                                img_h,
                                score_thresh=0.1,   # lower for debug
                                topk=20,
                                nms_iou=0.4
                            )

                        max_score = pred_scores_b.max().item() if pred_scores_b.numel() > 0 else 0.0
                        print(
                            f"[DET DEBUG] epoch={epoch} step={step} batch={b_idx} "
                            f"num_pred={pred_boxes_abs.shape[0]} num_gt={gt_boxes_b.shape[0]} "
                            f"max_score={max_score:.4f}"
                        )

                        save_path = f"det_debug_e{epoch}_s{step}_b{b_idx}.png"

                        show_det_debug(
                            img_tensor=imgs[b_idx],
                            pred_boxes=pred_boxes_abs,
                            pred_labels=pred_labels_b,
                            pred_scores=pred_scores_b,
                            gt_boxes=gt_boxes_b,
                            gt_labels=gt_labels_b,
                            det_class_names=getattr(train_loader.dataset, "det_class_names", None),
                            title=f"Epoch {epoch} | Step {step} | Batch {b_idx} | max_score={max_score:.4f}",
                            save_path=save_path,
                            show_window=True
                        )

                        print(f"[DET DEBUG] Saved: {save_path}")
                        # optional:
                        # print(f"[DET DEBUG] query_ids: {pred_query_ids_b}")
                        break  # only first detection sample from this step

                except Exception as e:
                    print(f"[DEBUG DET] Visualization error: {e}")
        
            if debug and (step % 100 == 0):
                print(f"\n[DEBUG] Step {step} Visualization")
                try:
                    # 1. Prepare Input Image
                    img_tensor = imgs[0].detach().cpu().permute(1, 2, 0).numpy()
                    # De-normalize
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_tensor + mean
                    img_np = np.clip(img_np, 0, 1)
                    
                    # ---------------------------------------------------------
                    # HELPER: Convert Tensor [C, H, W] -> Color Map [H, W]
                    # ---------------------------------------------------------
                    def tensor_to_colormap(mask_tensor):
                        # 1. Find the "Winner" Class Index (0, 1, 2...) for each pixel
                        class_indices = np.argmax(mask_tensor, axis=0)
                        
                        # 2. Shift to 1-based indices (1, 2, 3...) so 0 is background
                        color_map = class_indices + 1
                        max_val = np.max(mask_tensor, axis=0)
                        is_background = max_val < 0.5 
                        color_map[is_background] = 0
                        
                        return color_map
                    # ---------------------------------------------------------

                    plt.figure(figsize=(12, 6))
                    
                    # --- A. Ground Truth Plot ---
                    gt_map = None
                    if 'seg_mask' in batch:
                        gt_tensor = batch['seg_mask'][0].detach().cpu().numpy()
                        # Only generate map if we actually have data
                        if gt_tensor.max() > 0: 
                            gt_map = tensor_to_colormap(gt_tensor)
                    
                    ax1 = plt.subplot(1, 2, 1)
                    plt.imshow(img_np, cmap='gray')
                    
                    if gt_map is not None:
                        # Mask out 0s so background remains transparent (shows the MRI)
                        masked_gt = np.ma.masked_where(gt_map == 0, gt_map)
                        # 'tab20' gives distinct colors (Blue, Orange, Green, etc.)
                        plt.imshow(masked_gt, cmap='tab20', alpha=0.7, vmin=1, vmax=20)
                        plt.title("Ground Truth (Multi-Class)")
                    else:
                        plt.title("Ground Truth (Empty/Normal)")
                    plt.axis('off')

                    # --- B. Prediction Plot ---
                    pred_map = None
                    if outputs.get('pred_masks') is not None:
                        # Resize
                        pred_resized = nn.functional.interpolate(
                            outputs['pred_masks'][0:1], size=target_size, mode='bilinear', align_corners=False
                        )
                        # Sigmoid
                        pred_prob = torch.sigmoid(pred_resized[0]).detach().cpu().numpy()
                        pred_map = tensor_to_colormap(pred_prob)

                    ax2 = plt.subplot(1, 2, 2)
                    plt.imshow(img_np, cmap='gray')
                    if pred_map is not None:
                        masked_pred = np.ma.masked_where(pred_map == 0, pred_map)
                        plt.imshow(masked_pred, cmap='tab20', alpha=0.7, vmin=1, vmax=20)
                    plt.title("Prediction (Multi-Class)")
                    plt.axis('off')

                    plt.show()
                    
                except Exception as e:
                    print(f"Vis Error: {e}")
                    # ---------------------------
                # 2. Classification Print
                # ---------------------------
                if 'class_logits' in outputs and 'class_labels' in batch:
                    # Predictions
                    pred_probs = torch.sigmoid(outputs['class_logits'][0])
                    pred_c = (pred_probs > 0.5).int().tolist()
                    
                    # Ground Truth
                    gt_c = batch['class_labels'][0].int().tolist()
                    
                    # Map to Class Names
                    p_names = [train_ds.class_names[i] for i, v in enumerate(pred_c) if v]
                    g_names = [train_ds.class_names[i] for i, v in enumerate(gt_c) if v]
                    
                    print(f"  [CLS] GT: {g_names}")
                    print(f"  [CLS] Pred: {p_names}")

                # ---------------------------
                # 3. Report Print
                # ---------------------------
                if 'report_logits' in outputs and captions is not None:
                    # Find where the report actually starts (ignoring masked prompt)
                    lbls = report_labels[0]
                    valid_indices = (lbls != -100).nonzero(as_tuple=True)[0]
                    
                    if len(valid_indices) > 0:
                        start_idx = valid_indices[0] 
                        
                        # Slice logits/targets to show only the report part
                        valid_logits = outputs['report_logits'][0][start_idx:]
                        valid_gt_ids = captions[0][start_idx:]
                        
                        # Greedy Decode
                        pred_ids = valid_logits.argmax(dim=-1)
                        
                        p_txt = tokenizer.decode(pred_ids, skip_special_tokens=True)
                        g_txt = tokenizer.decode(valid_gt_ids, skip_special_tokens=True)
                        
                        print(f"  [REP] GT:   {g_txt}") # Truncate for cleaner log
                        print(f"  [REP] Pred: {p_txt}")
                    else:
                        print("  [REP] No valid labels found (All masked?)")
                            
        is_heavy_epoch = (epoch + 1 == 1) or ((epoch + 1) % 5 == 0) or (epoch + 1 == epochs)    
        if epoch % 2 == 0 and model.use_moe and model.task_moe is not None:
            task_names = ["seg", "cls", "rep", "det"]
            if getattr(model, "moe_type", None) == "sparse":
                bias_str = ", ".join(
                    f"{value:+.3f}" for value in model.task_moe.expert_bias.tolist()
                )
                gamma_str = ", ".join(
                    f"{task_names[i]}={value:.4f}"
                    for i, value in enumerate(model.task_moe.gamma.tolist())
                )
                print(f"[Sparse-MoE] expert bias: [{bias_str}]")
                print(f"[Sparse-MoE] gamma: [{gamma_str}]")
            elif getattr(model, "moe_type", None) == "gradif":
                _summary = model.task_moe.routing_summary()
                print(f"[FreqXpert-MoE][epoch={epoch}] routed frequency-expert mixtures")
                for _stage, _stage_info in _summary.get("stages", {}).items():
                    _dirs = _stage_info.get("directions", [])
                    _gamma = _stage_info.get("gamma", [])

                    # {"spatial": [...], "semantic": [...]}
                    if isinstance(_gamma, dict):
                        continue

                    _active_ids = _stage_info.get("active_task_ids", [])
                    _numeric_ids = []
                    for _id in _active_ids:
                        try:
                            _numeric_ids.append(int(_id))
                        except (TypeError, ValueError):
                            pass
                    _gamma_txt = ", ".join(
                        f"{task_names[i]}={float(_gamma[i]):.4f}"
                        for i in _numeric_ids
                        if 0 <= i < len(_gamma)
                    )
                    if _gamma_txt:
                        print(f"  [{_stage}] gamma: {_gamma_txt}")
                    else:
                        print(f"  [{_stage}] separate semantic routers")
                    _route_labels = {
                        "0": "seg", "1": "cls", "2": "rep", "3": "det",
                        "birads": "BI-RADS", "pathology": "pathology",
                        "classification": "generic-cls", "report": "report",
                    }
                    for _tid, _info in _stage_info.get("tasks", {}).items():
                        _weights = _info.get("mean_weights", [])
                        _pairs = ", ".join(
                            f"{_dirs[i]}={_weights[i]:.3f}"
                            for i in range(min(len(_dirs), len(_weights)))
                        )
                        print(
                            f"    {_route_labels.get(str(_tid), str(_tid))}: "
                            f"entropy={_info.get('normalized_entropy', 0.0):.3f}  {_pairs}"
                        )
                    if "mean_band_energy" in _stage_info:
                        print(f"    raw band energy: {_stage_info['mean_band_energy']}")
            elif getattr(model, "moe_type", None) == "evidence":
                print(f"[Evidence-MoE][epoch={epoch}] routing summary")
                for stage_name, bank in model.task_moe.items():
                    gamma_str = ", ".join(
                        f"{['seg','cls'][i]}={value:.3f}"
                        for i, value in enumerate(bank.gamma.detach().cpu().tolist())
                    )
                    print(f"  {stage_name}: gamma [{gamma_str}]")
                    for task_id, summary in bank.routing_summary().items():
                        usage = ", ".join(
                            f"{summary['expert_names'][i]}={value:.2f}"
                            for i, value in enumerate(summary['usage'])
                        )
                        print(
                            f"    {['seg','cls'][int(task_id)]}: "
                            f"entropy={summary['entropy']:.3f}  {usage}"
                        )


        # Validation & Saving
        RUN_VALIDATION=True
        avg_train = {k: np.mean(v) if v else 0.0 for k, v in train_stats.items()}
        if RUN_VALIDATION and val_loader:
            avg_val = validate(
                model, val_loader, device, loss_fns, manager, epoch,
                tokenizer=tokenizer, max_val_samples=None,
                run_heavy_metrics=is_heavy_epoch,
                roi_cls_weights=roi_cls_weights, pos_weight=pos_weight,
            )
            # Persist per-epoch validation metrics for analysis.
            try:
                _val_rec = {"epoch": int(epoch)}
                for _k, _v in avg_val.items():
                    try:
                        _val_rec[_k] = float(_v)
                    except (TypeError, ValueError):
                        pass
                RESULTS["val_history"].append(_val_rec)
                RESULTS["training_counters"] = {
                    "optimizer_steps": int(_optimizer_step_count),
                    "skipped_bad_gradient_steps": int(_skipped_bad_grad_steps),
                    "empty_gradient_windows": int(_empty_gradient_windows),
                }
                with open(_results_path, "w") as _rf:
                    json.dump(RESULTS, _rf, indent=2)
            except Exception as _e_res:
                print(f"[RESULTS] dump skipped: {_e_res}")
            # Task-family-balanced checkpoint criterion.  Each active clinical
            # task family contributes one normalized score, preventing dense
            # segmentation Dice from selecting a checkpoint with degraded
            # BI-RADS/pathology classification.
            _family_scores = {}

            _dice_values = [
                float(v) for k, v in avg_val.items()
                if k.endswith("_SegDice")
                and not k.endswith("_SegDice_lesion")
                and np.isfinite(v)
            ]
            if _dice_values:
                _family_scores["segmentation"] = float(np.mean(_dice_values))

            _global_cls_values = []
            for _key in ("Val_Macro_AUC", "Val_mAP"):
                _value = avg_val.get(_key)
                if _value is not None and np.isfinite(_value):
                    _global_cls_values.append(float(_value))
            if _global_cls_values:
                _family_scores["generic_classification"] = float(
                    np.mean(_global_cls_values)
                )

            _busbra_cls_values = []
            for _key in (
                "Val_BIRADS_QWK",
                "Val_BIRADS_MacroAUC_OvR",
                "Val_BIRADS_MacroF1",
                "Val_Pathology_AUC",
                "Val_Pathology_MacroF1",
            ):
                _value = avg_val.get(_key)
                if _value is not None and np.isfinite(_value):
                    _busbra_cls_values.append(float(_value))
            if _busbra_cls_values:
                _family_scores["busbra_classification"] = float(
                    np.mean(_busbra_cls_values)
                )

            _det_f1 = avg_val.get("Val_Det_F1")
            if _det_f1 is not None and np.isfinite(_det_f1) and float(_det_f1) > 0:
                _family_scores["detection"] = float(_det_f1)

            _roi_auc = avg_val.get("Val_LSS_MRI_ROI_CLS_MacroAUC")
            if _roi_auc is not None and np.isfinite(_roi_auc):
                _family_scores["mri_roi_classification"] = float(_roi_auc)

            _report_values = [
                float(value)
                for key, value in avg_val.items()
                if (
                    key.endswith("_ROUGE_L")
                    or key.endswith("_METEOR")
                )
                and np.isfinite(value)
            ]
            if _report_values:
                _family_scores["report_generation"] = float(
                    np.mean(_report_values)
                )

            if _family_scores:
                unified_score = float(np.mean(list(_family_scores.values())))
                avg_val["Val_UnifiedScore"] = unified_score
                monitor_loss = -unified_score
                _monitor_metric = "-Val_UnifiedScore"
                _score_report = ", ".join(
                    f"{name}={value:.4f}"
                    for name, value in _family_scores.items()
                )
                print(
                    f"  [Monitor] Task-family UnifiedScore={unified_score:.4f} "
                    f"from {_score_report}"
                )
            else:
                monitor_loss = avg_val.get("loss", float("inf"))
                _monitor_metric = "val_loss"
                print(f"  [Monitor] fallback Val loss={monitor_loss:.4f}")

            # Dedicated ultrasound checkpoint: generic BUS-BRA mAP is excluded.
            _us_components = {
                "dice": (0.25, avg_val.get("Val_BUS_BRA_SegDice")),
                "birads_qwk": (0.25, avg_val.get("Val_BIRADS_QWK")),
                "birads_auc": (0.20, avg_val.get("Val_BIRADS_MacroAUC_OvR")),
                "path_auc": (0.20, avg_val.get("Val_Pathology_AUC")),
                "path_f1": (0.10, avg_val.get("Val_Pathology_MacroF1")),
            }
            _us_sum = 0.0
            _us_weight = 0.0
            for _name, (_weight, _value) in _us_components.items():
                if _value is None:
                    continue
                try:
                    _value = float(_value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(_value):
                    _us_sum += _weight * _value
                    _us_weight += _weight
            if _us_weight > 0:
                ultrasound_score = _us_sum / _us_weight
                avg_val["Val_UltrasoundScore"] = float(ultrasound_score)
                _best_us = getattr(manager, "best_ultrasound_score", -float("inf"))
                if ultrasound_score > _best_us:
                    manager.best_ultrasound_score = float(ultrasound_score)
                    _us_path = os.path.join(
                        manager.ckpt_dir, "best_ultrasound_model.pth"
                    )
                    _us_tmp = _us_path + ".tmp"
                    torch.save(model.state_dict(), _us_tmp)
                    os.replace(_us_tmp, _us_path)
                    print(
                        f"  [+] New Best Ultrasound Model "
                        f"(score={ultrasound_score:.4f})"
                    )

            # Refresh the per-run artifact with the completed validation metrics.
            try:
                if RESULTS.get("val_history"):
                    for _k, _v in avg_val.items():
                        try:
                            RESULTS["val_history"][-1][_k] = float(_v)
                        except (TypeError, ValueError):
                            pass
                    RESULTS["training_counters"] = {
                        "optimizer_steps": int(_optimizer_step_count),
                        "skipped_bad_gradient_steps": int(_skipped_bad_grad_steps),
                        "empty_gradient_windows": int(_empty_gradient_windows),
                    }
                    with open(_results_path, "w") as _rf:
                        json.dump(RESULTS, _rf, indent=2)
            except Exception as _e_res:
                print(f"[RESULTS] GraDiF refresh skipped: {_e_res}")

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_loss)
                else:
                    scheduler.step()

            manager.log_epoch(epoch, avg_train, avg_val)
            saved_best = manager.save_checkpoint(
                model, optimizer, scheduler, scaler, monitor_loss, epoch,
                save_best=True, monitor_name=_monitor_metric
            )


        # del avg_val
        # torch.cuda.empty_cache()
        # gc.collect()


    if test_loader:
        _test_ckpt = manager.best_model_path
        print(f"Loading Best Model for Testing: {_test_ckpt}")
        state_dict = torch.load(_test_ckpt, map_location=device)
        model.load_state_dict(state_dict)
        # ---------------------------------------------------------
        # 1. Reset checkpoint thresholds to neutral CORAL thresholds
        # ---------------------------------------------------------
        core = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            print(
                "[BI-RADS] Stored thresholds:",
                core.birads_decode_thresholds.detach().cpu().tolist()
            )
            core.birads_decode_thresholds.fill_(0.5)
            print(
                "[BI-RADS] Reset to:",
                core.birads_decode_thresholds.detach().cpu().tolist()
            )
        # ---------------------------------------------------------
        # 2. Recalibrate ONCE using validation data
        # ---------------------------------------------------------
        print("[BI-RADS] Recalibrating once on validation set...")
        validate(
            model,
            val_loader,
            device,
            loss_fns,
            manager,
            epoch=101,  # selected checkpoint epoch
            tokenizer=tokenizer,
            max_val_samples=None,
            run_heavy_metrics=True,
            roi_cls_weights=roi_cls_weights,
            pos_weight=pos_weight,
        )

        print(
            "[BI-RADS] Final calibrated thresholds:",
            core.birads_decode_thresholds.detach().cpu().tolist()
        )
        # Optional: save separately; do not overwrite original checkpoint
        recalibrated_path = os.path.join(
            manager.ckpt_dir,
            "best_model_birads_recalibrated.pth"
        )
        torch.save(model.state_dict(), recalibrated_path)
        print(f"[Saved] {recalibrated_path}")
        test_multitask(
            model,
            test_loader,
            device,
            manager,
            tokenizer,
            pos_weight,
        )

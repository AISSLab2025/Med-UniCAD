import torch
import torch.nn as nn
import torch.nn.functional as F

from medunicad.losses.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def sanitize_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    boxes: [..., 4] in cx, cy, w, h
    returns sanitized cxcywh without in-place ops
    """
    boxes = torch.nan_to_num(boxes.float(), nan=0.5, posinf=1.0, neginf=0.0)
    cxcy = boxes[..., :2].clamp(0.0, 1.0)
    wh = boxes[..., 2:].clamp(min=1e-6, max=1.0)
    return torch.cat([cxcy, wh], dim=-1)


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha=0.25, gamma=2.0):
    """
    inputs:  [B, Q, C] or [N, C]
    targets: same shape, multi-hot
    """
    inputs = inputs.float()
    targets = targets.float()

    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
    prob = inputs.sigmoid().clamp(1e-6, 1.0 - 1e-6)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    denom = float(max(int(num_boxes), 1))
    if loss.dim() == 3:
        return loss.mean(dim=1).sum() / denom
    return loss.sum() / denom


class DINOCriterion(nn.Module):
    """
    DINO-style criterion:
      - focal classification loss
      - L1 box loss
      - GIoU loss
      - aux losses
      - DN losses
    """
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef=0.1,     # kept for compatibility; unused in focal mode
        use_focal=True,
        alpha=0.25,
        gamma=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma

    def _get_src_permutation_idx(self, indices, device):
        non_empty = [(src, tgt) for (src, tgt) in indices if src.numel() > 0]
        if len(non_empty) == 0:
            return (
                torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
            )

        batch_idx = torch.cat(
            [torch.full_like(src, i, device=device) for i, (src, _) in enumerate(indices) if src.numel() > 0],
            dim=0
        )
        src_idx = torch.cat([src.to(device) for (src, _) in indices if src.numel() > 0], dim=0)
        return batch_idx, src_idx

    def _num_boxes(self, targets):
        num_boxes = sum(len(t["labels"]) for t in targets)
        return max(num_boxes, 1)

    def _split_normal_outputs(self, outputs):
        dn_meta = outputs.get("dn_meta", None)
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]

        if dn_meta is None or dn_meta.get("num_dn_queries", 0) <= 0:
            outputs_wo_dn = {
                "pred_logits": pred_logits,
                "pred_boxes": pred_boxes,
            }
            if "aux_outputs" in outputs:
                outputs_wo_dn["aux_outputs"] = outputs["aux_outputs"]
            return outputs_wo_dn

        num_dn_queries = int(dn_meta["num_dn_queries"])

        outputs_wo_dn = {
            "pred_logits": pred_logits[:, num_dn_queries:],
            "pred_boxes": pred_boxes[:, num_dn_queries:],
        }

        if "aux_outputs" in outputs:
            aux_list = []
            for aux in outputs["aux_outputs"]:
                aux_list.append({
                    "pred_logits": aux["pred_logits"][:, num_dn_queries:],
                    "pred_boxes": aux["pred_boxes"][:, num_dn_queries:],
                })
            outputs_wo_dn["aux_outputs"] = aux_list

        return outputs_wo_dn

    def _sanitize_outputs(self, outputs):
        out = {
            "pred_logits": torch.nan_to_num(
                outputs["pred_logits"].float(), nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0),
            "pred_boxes": sanitize_cxcywh(outputs["pred_boxes"]),
        }

        if "aux_outputs" in outputs:
            aux_list = []
            for aux in outputs["aux_outputs"]:
                aux_logits = torch.nan_to_num(
                    aux["pred_logits"].float(), nan=0.0, posinf=20.0, neginf=-20.0
                ).clamp(-20.0, 20.0)
                aux_boxes = sanitize_cxcywh(aux["pred_boxes"])
                aux_list.append({"pred_logits": aux_logits, "pred_boxes": aux_boxes})
            out["aux_outputs"] = aux_list

        return out

    def _sanitize_targets(self, targets, device):
        clean_targets = []
        for t in targets:
            boxes = t["boxes"].to(device=device, dtype=torch.float32)
            labels = t["labels"].to(device=device, dtype=torch.long)

            if boxes.numel() == 0 or labels.numel() == 0:
                clean_targets.append({
                    "boxes": torch.zeros((0, 4), device=device, dtype=torch.float32),
                    "labels": torch.zeros((0,), device=device, dtype=torch.long),
                })
                continue

            boxes = sanitize_cxcywh(boxes)

            valid = torch.isfinite(boxes).all(dim=1)
            valid = valid & torch.isfinite(labels.float())
            valid = valid & (labels >= 0) & (labels < self.num_classes)
            valid = valid & (boxes[:, 2] > 1e-6) & (boxes[:, 3] > 1e-6)

            boxes = boxes[valid]
            labels = labels[valid]

            clean = {"boxes": boxes, "labels": labels}

            if "severity" in t:
                sev = t["severity"].to(device)
                if sev.numel() == valid.shape[0]:
                    clean["severity"] = sev[valid]
                else:
                    clean["severity"] = sev

            clean_targets.append(clean)

        return clean_targets

    def loss_labels(self, outputs, targets, indices, num_boxes):
        src_logits = outputs["pred_logits"]   # [B, Q, C]
        device = src_logits.device

        idx = self._get_src_permutation_idx(indices, device)
        target_classes_onehot = torch.zeros_like(src_logits)

        if idx[0].numel() > 0:
            target_classes_o = torch.cat(
                [t["labels"][J] for t, (_, J) in zip(targets, indices) if J.numel() > 0],
                dim=0
            )
            target_classes_onehot[idx[0], idx[1], target_classes_o] = 1.0

        loss_ce = sigmoid_focal_loss(
            src_logits,
            target_classes_onehot,
            num_boxes=num_boxes,
            alpha=self.alpha,
            gamma=self.gamma
        ) * src_logits.shape[1]

        return {"loss_ce": loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        device = outputs["pred_boxes"].device
        idx = self._get_src_permutation_idx(indices, device)

        if idx[0].numel() == 0:
            zero = outputs["pred_boxes"].sum() * 0.0
            return {"loss_bbox": zero, "loss_giou": zero}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [t["boxes"][J] for t, (_, J) in zip(targets, indices) if J.numel() > 0],
            dim=0
        )

        src_boxes = sanitize_cxcywh(src_boxes)
        target_boxes = sanitize_cxcywh(target_boxes)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum() / float(num_boxes)

        giou = generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)
        )
        loss_giou = (1.0 - torch.diag(giou)).sum() / float(num_boxes)

        return {"loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def get_loss(self, outputs, targets, indices, num_boxes):
        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def loss_dn(self, outputs, targets, dn_meta):
        if dn_meta is None:
            return {}

        num_dn_queries = int(dn_meta.get("num_dn_queries", 0))
        if num_dn_queries <= 0:
            return {}

        pred_logits = torch.nan_to_num(
            outputs["pred_logits"][:, :num_dn_queries].float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        pred_boxes = sanitize_cxcywh(outputs["pred_boxes"][:, :num_dn_queries])

        loss_ce_list = []
        loss_bbox_list = []
        loss_giou_list = []

        B = pred_logits.shape[0]
        device = pred_logits.device

        for b in range(B):
            pos_idx = dn_meta["dn_pos_idx"][b]
            gt_idx = dn_meta["dn_gt_idx"][b]

            if pos_idx.numel() == 0 or gt_idx.numel() == 0:
                continue

            tgt_labels = targets[b]["labels"][gt_idx]
            tgt_boxes = targets[b]["boxes"][gt_idx]

            valid = (tgt_labels >= 0) & (tgt_labels < self.num_classes)
            if valid.sum() == 0:
                continue

            pos_idx = pos_idx[valid]
            tgt_labels = tgt_labels[valid]
            tgt_boxes = tgt_boxes[valid]

            src_logits = pred_logits[b, pos_idx]
            src_boxes = pred_boxes[b, pos_idx]

            target_onehot = torch.zeros(
                (src_logits.shape[0], self.num_classes),
                device=device,
                dtype=src_logits.dtype
            )
            target_onehot.scatter_(1, tgt_labels.unsqueeze(1), 1.0)

            loss_ce_b = sigmoid_focal_loss(
                src_logits,
                target_onehot,
                num_boxes=max(src_logits.shape[0], 1),
                alpha=self.alpha,
                gamma=self.gamma
            ) * src_logits.shape[0]

            loss_bbox_b = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(tgt_boxes)
            )
            loss_giou_b = (1.0 - torch.diag(giou)).mean()

            loss_ce_list.append(loss_ce_b)
            loss_bbox_list.append(loss_bbox_b)
            loss_giou_list.append(loss_giou_b)

        if len(loss_ce_list) == 0:
            return {}

        return {
            "loss_dn_ce": torch.stack(loss_ce_list).mean(),
            "loss_dn_bbox": torch.stack(loss_bbox_list).mean(),
            "loss_dn_giou": torch.stack(loss_giou_list).mean(),
        }

    def forward(self, outputs, targets):
        device = outputs["pred_logits"].device
        targets = self._sanitize_targets(targets, device)

        outputs_wo_dn = self._split_normal_outputs(outputs)
        outputs_wo_dn = self._sanitize_outputs(outputs_wo_dn)

        outputs_main = {
            "pred_logits": outputs_wo_dn["pred_logits"],
            "pred_boxes": outputs_wo_dn["pred_boxes"],
        }

        indices = self.matcher(outputs_main, targets)
        num_boxes = self._num_boxes(targets)

        loss_dict = {}
        loss_dict.update(self.get_loss(outputs_main, targets, indices, num_boxes))

        if "aux_outputs" in outputs_wo_dn:
            for i, aux_outputs in enumerate(outputs_wo_dn["aux_outputs"]):
                need_aux = any(
                    f"{k}_{i}" in self.weight_dict
                    for k in ["loss_ce", "loss_bbox", "loss_giou"]
                )
                if not need_aux:
                    continue

                aux_indices = self.matcher(aux_outputs, targets)
                aux_loss_dict = self.get_loss(aux_outputs, targets, aux_indices, num_boxes)
                aux_loss_dict = {f"{k}_{i}": v for k, v in aux_loss_dict.items()}
                loss_dict.update(aux_loss_dict)

        # DN losses should be computed once, not inside the aux loop
        if "dn_meta" in outputs and outputs["dn_meta"] is not None:
            dn_losses = self.loss_dn(outputs, targets, outputs["dn_meta"])
            loss_dict.update(dn_losses)

        return loss_dict, indices

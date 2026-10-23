import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

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


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for DINO-style focal classification.

    outputs:
        pred_logits: [B, Q, C]      (NO no-object class)
        pred_boxes : [B, Q, 4]      normalized cxcywh
    targets:
        list of dicts with:
            labels: [N]
            boxes : [N, 4]          normalized cxcywh
    """
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
                 alpha=0.25, gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        pred_logits = torch.nan_to_num(
            outputs["pred_logits"].float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)

        pred_boxes = sanitize_cxcywh(outputs["pred_boxes"])

        bs, num_queries, num_classes = pred_logits.shape
        device = pred_logits.device

        if not torch.isfinite(pred_logits).all():
            bad = (~torch.isfinite(pred_logits)).sum().item()
            raise ValueError(f"[Matcher] pred_logits contains {bad} non-finite values")

        if not torch.isfinite(pred_boxes).all():
            bad = (~torch.isfinite(pred_boxes)).sum().item()
            raise ValueError(f"[Matcher] pred_boxes contains {bad} non-finite values")

        out_prob = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
        indices = []

        for b in range(bs):
            tgt_ids = targets[b]["labels"].long()
            tgt_bbox = targets[b]["boxes"].float()

            if tgt_ids.numel() == 0 or tgt_bbox.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device)
                ))
                continue

            tgt_bbox = sanitize_cxcywh(tgt_bbox)

            valid = torch.isfinite(tgt_bbox).all(dim=1)
            valid = valid & torch.isfinite(tgt_ids.float())
            valid = valid & (tgt_ids >= 0) & (tgt_ids < num_classes)

            tgt_ids = tgt_ids[valid]
            tgt_bbox = tgt_bbox[valid]

            if tgt_ids.numel() == 0 or tgt_bbox.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.int64, device=device)
                ))
                continue

            p = out_prob[b][:, tgt_ids]   # [Q, Nt]

            neg_cost = (1.0 - self.alpha) * (p ** self.gamma) * (-(1.0 - p).log())
            pos_cost = self.alpha * ((1.0 - p) ** self.gamma) * (-(p).log())
            cost_class = pos_cost - neg_cost

            cost_bbox = torch.cdist(pred_boxes[b], tgt_bbox, p=1)

            pred_xyxy = box_cxcywh_to_xyxy(pred_boxes[b])
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_bbox)
            cost_giou = -generalized_box_iou(pred_xyxy, tgt_xyxy)

            C = (
                self.cost_class * cost_class +
                self.cost_bbox * cost_bbox +
                self.cost_giou * cost_giou
            )

            if not torch.isfinite(cost_class).all():
                raise ValueError(f"[Matcher] cost_class has non-finite values in batch item {b}")
            if not torch.isfinite(cost_bbox).all():
                raise ValueError(f"[Matcher] cost_bbox has non-finite values in batch item {b}")
            if not torch.isfinite(cost_giou).all():
                raise ValueError(f"[Matcher] cost_giou has non-finite values in batch item {b}")
            if not torch.isfinite(C).all():
                raise ValueError(f"[Matcher] final cost matrix C has non-finite values in batch item {b}")

            C_np = C.detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_np)

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64, device=device),
                torch.as_tensor(col_ind, dtype=torch.int64, device=device)
            ))

        return indices

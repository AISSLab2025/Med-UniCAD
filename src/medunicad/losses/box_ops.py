import torch


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    x0 = torch.minimum(boxes[..., 0], boxes[..., 2])
    y0 = torch.minimum(boxes[..., 1], boxes[..., 3])
    x1 = torch.maximum(boxes[..., 0], boxes[..., 2])
    y1 = torch.maximum(boxes[..., 1], boxes[..., 3])
    return torch.stack([x0, y0, x1, y1], dim=-1)


def _sanitize_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=0.0)
    h = h.clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)


def box_cxcywh_to_xyxy(x):
    x = _sanitize_cxcywh(x)
    cx, cy, w, h = x.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


def box_xyxy_to_cxcywh(x):
    x = _sanitize_xyxy(x)
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([
        (x0 + x1) / 2.0,
        (y0 + y1) / 2.0,
        (x1 - x0).clamp(min=0.0),
        (y1 - y0).clamp(min=0.0)
    ], dim=-1)


def box_area(boxes):
    boxes = _sanitize_xyxy(boxes)
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0.0) * \
           (boxes[..., 3] - boxes[..., 1]).clamp(min=0.0)


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return (
            torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype),
            torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device, dtype=boxes1.dtype),
        )

    boxes1 = _sanitize_xyxy(boxes1)
    boxes2 = _sanitize_xyxy(boxes2)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])   # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])   # [N,M,2]

    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]

    union = (area1[:, None] + area2 - inter).clamp(min=1e-6)
    iou = inter / union
    iou = torch.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]),
            dtype=boxes1.dtype,
            device=boxes1.device
        )

    b1 = _sanitize_xyxy(boxes1)
    b2 = _sanitize_xyxy(boxes2)

    inter_xmin = torch.max(b1[:, None, 0], b2[:, 0])
    inter_ymin = torch.max(b1[:, None, 1], b2[:, 1])
    inter_xmax = torch.min(b1[:, None, 2], b2[:, 2])
    inter_ymax = torch.min(b1[:, None, 3], b2[:, 3])

    inter = (inter_xmax - inter_xmin).clamp(min=0) * \
            (inter_ymax - inter_ymin).clamp(min=0)

    area1 = (b1[:, 2] - b1[:, 0]).clamp(min=0) * (b1[:, 3] - b1[:, 1]).clamp(min=0)
    area2 = (b2[:, 2] - b2[:, 0]).clamp(min=0) * (b2[:, 3] - b2[:, 1]).clamp(min=0)

    union = (area1[:, None] + area2 - inter).clamp(min=1e-6)
    iou = inter / union

    x_min = torch.min(b1[:, None, 0], b2[:, 0])
    y_min = torch.min(b1[:, None, 1], b2[:, 1])
    x_max = torch.max(b1[:, None, 2], b2[:, 2])
    y_max = torch.max(b1[:, None, 3], b2[:, 3])

    area_c = ((x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)).clamp(min=1e-6)

    giou = iou - (area_c - union) / area_c
    giou = torch.nan_to_num(giou, nan=0.0, posinf=0.0, neginf=0.0)
    return giou


def masks_to_boxes(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float32)

    n, h, w = masks.shape
    y = torch.arange(h, device=masks.device, dtype=torch.float32)
    x = torch.arange(w, device=masks.device, dtype=torch.float32)

    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    x_mask = masks * x_grid.unsqueeze(0)
    y_mask = masks * y_grid.unsqueeze(0)

    x_max = x_mask.flatten(1).max(dim=1).values
    y_max = y_mask.flatten(1).max(dim=1).values

    x_min = x_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(dim=1).values
    y_min = y_mask.masked_fill(~masks.bool(), float("inf")).flatten(1).min(dim=1).values

    empty = masks.flatten(1).sum(dim=1) == 0
    x_min[empty] = 0
    y_min[empty] = 0
    x_max[empty] = 0
    y_max[empty] = 0

    return torch.stack([x_min, y_min, x_max, y_max], dim=1)


def normalize_boxes(boxes_xyxy, img_w, img_h):
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0, 4), device=boxes_xyxy.device, dtype=boxes_xyxy.dtype)

    img_w = max(float(img_w), 1.0)
    img_h = max(float(img_h), 1.0)

    boxes = _sanitize_xyxy(boxes_xyxy.clone())
    boxes[:, 0::2] = boxes[:, 0::2] / img_w
    boxes[:, 1::2] = boxes[:, 1::2] / img_h
    boxes = box_xyxy_to_cxcywh(boxes)
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    boxes[:, :2] = boxes[:, :2].clamp(0.0, 1.0)
    boxes[:, 2:] = boxes[:, 2:].clamp(0.0, 1.0)
    return boxes


def denormalize_boxes(boxes_cxcywh, img_w, img_h):
    if boxes_cxcywh.numel() == 0:
        return torch.zeros((0, 4), device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype)

    img_w = max(float(img_w), 1.0)
    img_h = max(float(img_h), 1.0)

    boxes = torch.nan_to_num(boxes_cxcywh.clone(), nan=0.0, posinf=1.0, neginf=0.0)
    boxes[:, :2] = boxes[:, :2].clamp(0.0, 1.0)
    boxes[:, 2:] = boxes[:, 2:].clamp(0.0, 1.0)

    boxes = box_cxcywh_to_xyxy(boxes)
    boxes[:, 0::2] *= img_w
    boxes[:, 1::2] *= img_h
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_w - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_h - 1)
    return boxes
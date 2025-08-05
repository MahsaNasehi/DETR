from torchvision.ops import generalized_box_iou as torch_giou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import torch

# def box_cxcywh_to_xyxy(x):
#     # Convert bounding box format from center_x, center_y, width, height
#     # to x_min, y_min, x_max, y_max
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

# def generalized_box_iou(boxes1, boxes2):
#     # Compute IoU (Intersection over Union) between two sets of boxes
#     # boxes are expected in [x_min, y_min, x_max, y_max] format
#     # This is a simplified version just for illustration

#     # Intersection box
#     x_min = torch.max(boxes1[:, None, 0], boxes2[:, 0])
#     y_min = torch.max(boxes1[:, None, 1], boxes2[:, 1])
#     x_max = torch.min(boxes1[:, None, 2], boxes2[:, 2])
#     y_max = torch.min(boxes1[:, None, 3], boxes2[:, 3])

#     # clamp(min=0) is a function in PyTorch (and many other libraries) that limits (or “clamps”) values to be at least a minimum value—in this case, 0.

#     inter = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)

#     area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
#     area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

#     union = area1[:, None] + area2 - inter

#     iou = inter / union

#     # generalized IoU: IoU - (enclosing_area - union)/enclosing_area (skipped for simplicity)
#     return iou

def hungarian_match(pred_logits, pred_boxes, target_labels, target_boxes):
    # (no batch for simplicity)
    """
    Args:
        pred_logits: Tensor [num_queries, num_classes]
        pred_boxes: Tensor [num_queries, 4] in [cx, cy, w, h]
        target_labels: Tensor [num_targets]
        target_boxes: Tensor [num_targets, 4]

    Returns:
        row_ind: indices in predictions
        col_ind: indices in targets
    """

    # 1. Compute classification cost
    pred_probs = pred_logits.softmax(-1)  # [num_queries, num_classes]
    # print("pred_logits.shape:", pred_logits.shape)

    # cost_class = -pred_probs[:, target_labels]  # negative prob for target class
    cost_class = -pred_probs.gather(1, target_labels[None].repeat(pred_probs.shape[0], 1))

    # 2. Compute bbox cost: L1 distance
    # cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)

    # if target_boxes.ndim == 1:
    #   target_boxes = target_boxes.unsqueeze(0)  # shape becomes [1, 4]
    # print(f"pred_boxes {pred_boxes.shape}")
    # print(f"trg ah: {target_boxes.shape}")
    cost_bbox = torch.cdist(pred_boxes, target_boxes)
    # Convert to [x1, y1, x2, y2]
    # pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
    # target_xyxy = box_cxcywh_to_xyxy(target_boxes)

    # Compute GIoU cost (distance)
    iou = torch_giou(pred_boxes, target_boxes)  # [num_queries, num_targets]
    cost_giou = -iou  # higher iou → lower cost

    # Total cost matrix (weights can be tuned)
    cost_matrix = 1.0 * cost_class + 5.0 * cost_bbox + 2.0 * cost_giou
    cost_matrix = cost_matrix.nan_to_num(posinf=1e9, neginf=-1e9)

    # Hungarian matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu())
    # indices of the rows (typically predictions), indices of the columns (typically ground truths)
    return row_ind, col_ind

def detr_loss(pred_logits, pred_boxes, target_labels, target_boxes):
    if target_labels.numel() == 0:
        # No objects in this image
        # You can still compute classification loss vs. "no object" class
        # Or just return 0.0
        return torch.tensor(0.0, device=pred_logits.device)
    λ_cls = 1.0
    λ_bbox = 5.0
    λ_giou = 2.0

    # Get matches
    indices_pred, indices_target = hungarian_match(pred_logits, pred_boxes, target_labels, target_boxes)

    # Select matched predictions and targets
    matched_pred_logits = pred_logits[indices_pred]
    matched_pred_boxes = pred_boxes[indices_pred]
    # These lines reorder the ground truth labels and boxes to match the order of the predictions.
    matched_target_labels = target_labels[indices_target]
    matched_target_boxes = target_boxes[indices_target]

    # Classification loss (cross-entropy)
    cls_loss = F.cross_entropy(matched_pred_logits, matched_target_labels)

    # Box regression loss (L1 loss)
    bbox_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes)

    # GIoU loss
    # pred_boxes_xyxy = box_cxcywh_to_xyxy(matched_pred_boxes)
    # target_boxes_xyxy = box_cxcywh_to_xyxy(matched_target_boxes)

    giou = torch_giou(matched_pred_boxes, matched_target_boxes)
    giou_loss = 1.0 - giou.diag().mean()

    # Total loss with weights
    total_loss = λ_cls * cls_loss + λ_bbox * bbox_loss + λ_giou * giou_loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("NaN or Inf detected in loss!")
        print("pred_logits:", pred_logits)
        print("pred_boxes:", pred_boxes)
        print("target_labels:", target_labels)
        print("target_boxes:", target_boxes)
        # exit()
    if total_loss.item() < 0:
        print("Negative loss detected!")
        print("pred:", indices_pred)
        print("tgt:", indices_target)
        print("giou_loss:", giou_loss.item())
        print("Pred boxes:", matched_pred_boxes)
        print("Tgt boxes:", matched_target_boxes)


    
    return total_loss


def compute_loss(outputs, targets, device):
    """
    outputs: dict with
       'pred_logits': [batch_size, num_queries, num_classes]
       'pred_boxes': [batch_size, num_queries, 4]
    targets: list of dicts, length = batch_size
        each dict has keys 'labels' (tensor) and 'boxes' (tensor)
    """
    batch_size = outputs['pred_logits'].shape[0]
    losses = []

    for i in range(batch_size):
        target_labels = targets[i]['labels'] - 1  # Make sure this is a tensor
        loss_i = detr_loss(
            outputs['pred_logits'][i],  # [num_queries, num_classes]
            outputs['pred_boxes'][i],   # [num_queries, 4]
            target_labels.to(device),
            targets[i]['boxes'].to(device)
        )
        losses.append(loss_i)

    return torch.stack(losses).mean()


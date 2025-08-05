import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import config
from models.detr import get_model

# Load and preprocess image
def load_image(image_path):
    transform = config.get_transform()
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0), image  # (1, C, H, W), original PIL image

# Convert bounding boxes from cxcywh to xmin, ymin, xmax, ymax
def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

# Draw boxes on image
def draw_boxes(image, boxes, scores, labels, threshold=0.5):
    draw = ImageDraw.Draw(image)
    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text((x_min, y_min), f"{label}: {score:.2f}", fill="white")
    return image

# Run inference
def run_inference(image_path, model_path):
    # Load model
    args = config.get_args()
    model = get_model(num_classes=args['num_classes'], num_queries=args['num_query'], d_model=args['d_model'], device=args['device'])
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load image
    tensor_img, pil_img = load_image(image_path)

    # Run model
    with torch.no_grad():
        outputs = model(tensor_img.to(args['device']))
    
    # Get outputs
    logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    boxes = outputs['pred_boxes'][0]    # [num_queries, 4] in cxcywh normalized

    # Get class scores (excluding "no-object" class) and get the max per box
    probs = logits.softmax(-1)[:, :-1]         # shape [num_queries, num_classes]
    scores, labels = probs.max(-1)             # Get max score and corresponding class index

    # Filter boxes with score > threshold
    keep = scores > 0.5
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]  
    # Unnormalize boxes to image size
    w, h = pil_img.size
    
    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes.device)

    # Draw boxes
    result_img = draw_boxes(pil_img, boxes, scores, labels)
    result_img.show()
    # Optionally save the image
    result_img.save("output.jpg")

if __name__ == "__main__":
    run_inference("sheep.jpg", "saved_model.pth")

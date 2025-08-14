from PIL import Image
import config
import torch
import json
import os


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, annotations_file, transforms=None) -> None:
        """
        Dataset for Image Data.

        Args:
            data_dir (str): Directory containing the image files.
            annotations_file (str): Path to the annotations file for supervised tasks.
            transforms (callable, optional): Optional transformations to be applied on images.
        """
        self.img_dir = data_dir
        with open(annotations_file) as f:
            coco = json.load(f)
        self.transforms = transforms
        self.images = coco["images"]
        self.annotations = coco["annotations"]
        self.categories = coco["categories"]

        # Build image_id -> annotations mapping
        self.img_id_to_ann = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.img_id_to_ann:
                self.img_id_to_ann[img_id] = []
            self.img_id_to_ann[img_id].append(ann)

        # Optional: map category_id to category name (if needed)
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.categories}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_filename = img_info["file_name"]
        img_path = os.path.join(self.img_dir, img_filename)

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        # Get annotations for this image
        anns = self.img_id_to_ann.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x_min = x / orig_w
            y_min = y / orig_h
            x_max = (x + w) / orig_w
            y_max = (y + h) / orig_h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])
        if len(boxes) == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        # for ann in anns:
        #   x, y, w, h = ann['bbox']
        #   boxes.append([x, y, x + w, y + h])
        #   labels.append(ann['category_id'])

        # Convert to tensor
        # boxes = torch.tensor(boxes, dtype=torch.float32)
        # labels = torch.tensor(labels, dtype=torch.int64)

        if self.transforms:
            image = self.transforms(image)
            # new_w, new_h = image.shape[2], image.shape[1]
            # scale_x = new_w / orig_w
            # scale_y = new_h / orig_h
            # boxes[:, [0, 2]] *= scale_x
            # boxes[:, [1, 3]] *= scale_y

        target = {
            "boxes": boxes,  # normalized
            "labels": labels,
            "image_id": torch.tensor([img_id]),
        }

        return image, target


def collate_fn(batch):
    images = [item[0] for item in batch]  # list of image tensors
    targets = [item[1] for item in batch]  # list of dicts (variable sizes)

    images = torch.stack(images, dim=0)  # stack images to tensor [B, C, H, W]

    return images, targets


def get_dataset(data_dir, annotations_file, transform):
    dataset = CustomDataset(
        data_dir=data_dir, annotations_file=annotations_file, transforms=transform
    )
    return dataset

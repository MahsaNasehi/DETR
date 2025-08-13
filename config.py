from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_names():
    return {
        0: "VOC",
        1: "aeroplane",
        2: "bicycle",
        3: "bird",
        4: "boat",
        5: "bottle",
        6: "bus",
        7: "car",
        8: "cat",
        9: "chair",
        10: "cow",
        11: "diningtable",
        12: "dog",
        13: "horse",
        14: "motorbike",
        15: "person",
        16: "pottedplant",
        17: "sheep",
        18: "sofa",
        19: "train",
        20: "tvmonitor",
    }


def get_args():
    return {
        "batch_size": 4,
        "lr": 1e-5,
        "epochs": 100,
        "device": device,
        "num_query": 100,
        "num_classes": 20,
        "d_model": 256,
        "data_path": "/data/Pascal-VOC-2012-1/train",
        "annotation_file": "/data/Pascal-VOC-2012-1/train/_annotations.coco.json",
    }


def get_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

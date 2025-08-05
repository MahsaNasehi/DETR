# DETR Object Detection (PyTorch)
This project is a modular and simplified implementation of Facebook AI’s DETR (DEtection TRansformer) model, adapted for educational and research purposes. DETR formulates object detection as a direct set prediction problem using a transformer-based architecture, eliminating the need for hand-crafted components like anchor boxes and NMS (non-maximum suppression).
-------------------------------------------------------------------------------
The codebase is written in pure Python with PyTorch and supports:
* Training on custom datasets using standard COCO-style annotations

* Hungarian matching loss with L1 and GIoU components

* Inference on local images with visualization of predicted bounding boxes

* Modular structure for extending loss functions or backbone architectures

The model outputs bounding boxes in (x_min, y_min, x_max, y_max) format normalized to [0, 1], and predictions are matched to targets via bipartite matching (Hungarian algorithm).

To run inference or fine-tune on your own dataset, check the main.py and test.py scripts.

## Custom Dataset
* Implements a PyTorch Dataset for loading images and COCO-format annotations.

* Initialization:

data_dir: folder with images.

annotations_file: JSON with annotations.

transforms: optional image transformations.

* Parses annotations and builds a mapping from image IDs to their annotations.

* __getitem__:

Loads an image by index.

Retrieves and normalizes bounding boxes relative to image size.

Returns transformed image tensor and target dictionary with:

boxes: normalized bounding boxes.

labels: category IDs.

image_id: tensor with image identifier.

* Supports batching with a custom collate_fn to stack images and keep targets as lists (for variable-length annotations).

## DETR Loss & Hungarian Matching
* Hungarian Matching:

  * Matches predicted boxes and classes to ground truth using a cost matrix combining:

    * Classification cost (negative predicted class probabilities),
    
    * Bounding box L1 distance,
    
    * Generalized IoU (GIoU) cost.

  * Solves assignment using the Hungarian algorithm (linear_sum_assignment).

* Loss Computation:

  * Uses matched pairs to compute:

    * Classification loss (cross-entropy),

    * Bounding box regression loss (L1),

GIoU loss.

  * Combines these with weights (e.g., 1.0 for class, 5.0 for bbox, 2.0 for GIoU).

  * Handles batches by computing loss per example and averaging.

# DETR Object Detection (PyTorch)
This project is a modular and simplified implementation of Facebook AI’s DETR (DEtection TRansformer) model, adapted for educational and research purposes. DETR formulates object detection as a direct set prediction problem using a transformer-based architecture, eliminating the need for hand-crafted components like anchor boxes and NMS (non-maximum suppression).

The codebase is written in pure Python with PyTorch and supports:
* Training on custom datasets using standard COCO-style annotations

* Hungarian matching loss with L1 and GIoU components

* Inference on local images with visualization of predicted bounding boxes

* Modular structure for extending loss functions or backbone architectures

The model outputs bounding boxes in (x_min, y_min, x_max, y_max) format normalized to [0, 1], and predictions are matched to targets via bipartite matching (Hungarian algorithm).

To run inference or fine-tune on your own dataset, check the main.py and test.py scripts.

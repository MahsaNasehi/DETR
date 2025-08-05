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

 * data_dir: folder with images.
 
 * annotations_file: JSON with annotations.
 
 * transforms: optional image transformations.

* Parses annotations and builds a mapping from image IDs to their annotations.

* __getitem__:

 * Loads an image by index.
 
 * Retrieves and normalizes bounding boxes relative to image size.
 
 * Returns transformed image tensor and target dictionary with:
 
 * boxes: normalized bounding boxes.
 
 * labels: category IDs.
 
 * image_id: tensor with image identifier.

* Supports batching with a custom collate_fn to stack images and keep targets as lists (for variable-length annotations).

## DETR Model Components
### 1. PositionEncodingSineCosine
 * Generates sine-cosine positional encodings for 2D feature maps.
 
 * Input: mask tensor [B, H, W] (True for padded pixels).
 
 * Output: positional embedding [B, 2*num_pos_feats, H, W].
 
 * Helps the transformer know spatial positions without learned embeddings.

### 2. BackboneWithPE
 * Wraps a CNN backbone (e.g., ResNet-50) without the final pooling and FC layers.
 
 * Extracts feature maps [B, 2048, H/32, W/32].
 
 * Generates a zero mask and applies positional encoding.
 
 * Returns: (features, mask, pos_encoding).

### 3. Transformer
 * Transformer encoder-decoder stack built with PyTorch's nn.TransformerEncoder and nn.TransformerDecoder.
 
 * Inputs:
 
  * src: Flattened feature map + positional encoding.
  
  * mask: padding mask for the encoder.
  
  * query_embed: learnable object queries.
  
  * pos_embed: positional encoding for input.
 
 * Outputs the decoder hidden states for object queries.

### 4. DETR (Detection Transformer)
 * Integrates backbone, transformer, and prediction heads.
 
 * Key modules:
 
  * query_embed: learnable object queries for decoder input.
  
  * input_proj: 1x1 conv to project backbone features to d_model.
  
  * class_embed: linear layer to classify each query + "no-object" class.
  
  * bbox_embed: MLP to regress bounding boxes in [cx, cy, w, h] format.
 
 * Forward pass:
 
  * Extract features and positional embeddings.
  
  * Flatten features/masks for transformer.
  
  * Run transformer encoder-decoder.
  
  * Predict class logits and box coordinates.
  
  * Convert predicted boxes to [x_min, y_min, x_max, y_max] and clamp to [0,1].
 
 * Returns predictions dict with:
  
  * 'pred_logits': classification scores [B, num_queries, num_classes+1]
  
  * 'pred_boxes': bounding boxes [B, num_queries, 4]

### 5. MLP
 * Simple multi-layer perceptron for box regression.
 
 * Configurable layers, hidden dims, and output dims.


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

    * GIoU loss.

  * Combines these with weights (e.g., 1.0 for class, 5.0 for bbox, 2.0 for GIoU).

  * Handles batches by computing loss per example and averaging.

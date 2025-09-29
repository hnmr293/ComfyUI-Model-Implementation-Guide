# ComfyUI Custom Model Implementation Guide

This guide describes the procedures for defining and implementing new models on ComfyUI.

## 1. Custom Model Implementation

Please refer to [./Custom_Model_Guide.md](Custom_Model_Guide.md).
This describes the procedure for enabling custom models to be loaded from standard CheckpointLoader and similar nodes.

## 2. Custom Text Encoder Implementation

Please refer to [./Custom_ClipTarget_Guide.md](Custom_ClipTarget_Guide.md).
This describes the procedure for implementing custom text encoders.

## 3. Supplementary Information

When implementing according to this guide, the overall directory structure will be as follows:

```
custom_nodes/   # ComfyUI custom nodes directory
 └── my_model/  # Root directory for this custom node
      ├── utils/
      |    └── model_detection.py   # Model detection hook
      ├── models/
      |    ├── supported_models.py  # Model implementation
      |    ├── model_base.py        # Model implementation
      |    ├── comfy_intf.py        # Model implementation
      |    ├── model.py             # Model implementation
      |    ├── clip_comfy_intf.py   # TE implementation (only if needed)
      |    └── clip_model.py        # TE implementation (only if needed)
      ├── nodes.py                  # TE loader node (only if needed)
      └── __init__.py
```
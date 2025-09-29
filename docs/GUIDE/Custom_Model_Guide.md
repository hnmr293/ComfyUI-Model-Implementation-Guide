# ComfyUI Custom Model Implementation Guide

## Overview

This guide explains how to implement custom models in ComfyUI. It covers how to load models from existing UNETLoader or CheckpointLoaderSimple and integrate them with ComfyUI's model system architecture.

## Architecture Overview

### Model Loading Flow

```
UNETLoader / CheckpointLoaderSimple
    ↓
comfy.sd.load_diffusion_model / load_checkpoint_guess_config
    ↓
load_diffusion_model_state_dict
    ├── model_detection.detect_unet_config  [Extract model info from state_dict]
    ├── model_config_from_unet_config  [Search for matching model from supported_models]
    └── BASE.get_model  [Generate BaseModel instance]
        ↓
    ModelPatcher  [Final model wrapper]
```

## Required Components for Implementation

### Example Directory Structure

```
custom_nodes/
 └── my_model/
      ├── utils/
      |    └── model_detection.py
      ├── models/
      |    ├── supported_models.py
      |    ├── model_base.py
      |    ├── comfy_intf.py
      |    └── model.py  # <- Assume actual model implementation is here
      └── __init__.py
```

### 1. Model Detection System (`model_detection.py`)

The existing `model_detection.detect_unet_config` determines the model type and configuration values from the weight file contents (key names). We hook this process to properly detect the model we're implementing.

The return value of `model_detection.detect_unet_config` is a dictionary. Looking ahead to the next section, among classes that inherit from `supported_models.BASE`, the model whose properties match this dictionary's contents is selected as the corresponding model.

```python
# utils/model_detection.py

import comfy.model_detection

_orig_detect_unet_config = comfy.model_detection.detect_unet_config

def apply_custom_detection_patch() -> None:
    """
    Add hooks to ComfyUI's model detection.
    Call this from __init__.py etc. to enable custom model detection.
    Note that handling for multiple calls should be implemented (omitted here for brevity).
    """
    comfy.model_detection.detect_unet_config = _detect_unet_config

def _detect_custom_model(state_dict, key_prefix):
    """Custom model detection logic"""
    # Check for model-specific keys
    if f'{key_prefix}unique_layer.weight' not in state_dict:
        # It was a different model
        return None

    # Looks like custom model weights, so extract model configuration
    config = {
        'image_model': 'my_custom_model',
        'depth': len(key for key in state_dict if 'self_attn.to_q.weight' in key),
        'dim': state_dict[f'{key_prefix}embedder.weight'].size(0),
        'in_channels': 16,
        # Other necessary settings...
    }

    return config

def _detect_unet_config(state_dict, key_prefix, *args, **kwargs):
    """Hook detect_unet_config"""
    # Should call original implementation first to avoid breaking existing workflows
    # However, if structure is similar to existing models and cannot be distinguished, handle flexibly
    unet_config = _orig_detect_unet_config(state_dict, key_prefix, *args, **kwargs)

    # If no existing model is found, call custom model detection logic
    # If similar to existing models and misdetection occurs, you can check model name with `unet_config.get('image_model')` for special processing
    if unet_config is None:
        custom_config = _detect_custom_model(state_dict, key_prefix)
        if custom_config is not None:
            unet_config = custom_config

    return unet_config
```

### 2. BASE Definition (`supported_models.py`)

Configuration class for integration with ComfyUI's model system. The model is selected by comparing the dictionary returned by `model_detection.detect_unet_config` with this class's properties. Properties just need to be retrievable from the instance. They can be either class variables or instance variables. The following methods need to be defined:
- `get_model`: Should return the model for inference (`BaseModel`).
- `clip_target`: Should return the text encoder (`ClipTarget`).

```python
# models/supported_models.py

from comfy import supported_models_base, latent_formats
import torch

class MyModel(supported_models_base.BASE):
    """Custom model support definition"""

    # Configuration for model identification
    unet_config = {
        'image_model': 'my_custom_model',
    }

    # Required keys (keys that should exist in state_dict)
    required_keys = []

    # Memory usage coefficient
    # Not well understood (>_<)
    memory_usage_factor = 2.8

    # Supported inference data types
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    # Latent format
    latent_format = latent_formats.Flux()

    def __init__(self, unet_config):
        super().__init__(unet_config)
        # If you want to configure based on unet_config contents, do it here
        #   self.latent_format = ...
        # etc.

    def get_model(self, state_dict, prefix='', device=None):
        """Generate BaseModel instance"""
        return MyModelBase(self, device=device)

    @property
    def vae_key_prefix(self) -> list[str]:
        """
        State_dict prefix for VAE (if necessary)
        Can be instance variable or class variable
        """
        return ['autoencoder.']

    @property
    def text_encoder_key_prefix(self) -> list[str]:
        """
        State_dict prefix for text encoder (if necessary)
        Can be instance variable or class variable
        """
        return ['text_encoder.']

    def clip_target(self, state_dict={}) -> supported_models_base.ClipTarget:
        """Return the CLIP/text encoder used by the custom model"""
        # Determine from state_dict and return Tokenizer and EncoderModel
        from comfy.text_encoders import sd3_clip, flux
        prefix = self.text_encoder_key_prefix[0]
        t5_detect = sd3_clip.t5_xxl_detect(state_dict, f"{prefix}t5xxl.transformer.")
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))
```

### 3. BaseModel Implementation (`model_base.py`)

Wrapper class for the actual model processing. Created by inheriting from `model_base.BaseModel`. Pass the torch model class to the `BaseModel` constructor argument `unet_model`. The generated model is set to `self.diffusion_model`.

The key names in `state_dict` and the argument `unet_prefix` differ depending on the node that initiated the loading, so handle them appropriately as needed.

```python
# models/model_base.py

from torch import Tensor
from comfy import model_base
import comfy.conds
from .supported_models import MyModel
from .comfy_intf import MyModelIntf
from typing import Any

class MyModelBase(model_base.BaseModel):
    """
    Wrapper class for custom model.
    The actual model is stored in BaseModel.diffusion_model.
    """

    def __init__(
        self,
        model_config: MyModel,         # The BASE we created earlier
        model_type=model_base.ModelType.EPS, # Choose appropriate value from model_base.ModelType enumeration
        device=None,
    ):
        # Pass implementation class to unet_model parameter
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=MyModelIntf,
        )

    def extra_conds(self, **kwargs) -> dict[str, Any]:
        """
        Prepare what you want to pass as keyword arguments to custom model's forward.
        Text encoder output is passed in kwargs.
        """
        out = super().extra_conds(**kwargs)

        # As an example, let's add attention_mask
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        return out

    def load_model_weights(self, sd: dict[str, Tensor], unet_prefix: str = '') -> "MyModelBase":
        """Load model weights"""
        # Process sd key names appropriately as needed.

        # sd key names and unet_prefix vary depending on the node that initiated the loading
        # - When loaded from CheckpointLoader, unet_prefix is automatically determined as one of 'model.', 'net.', 'model.model.', 'model.diffusion_model.'. sd is passed as is from loading.
        # - When loaded from UNETLoader, unet_prefix is empty string. sd is passed with automatically determined prefix removed.

        # BaseModel.load_model_weights:
        # 1. Collects keys starting with unet_prefix from the passed sd (if unet_prefix is empty, all elements are used)
        # 2. Removes unet_prefix from collected keys
        # 3. Calls self.diffusion_model.load_state_dict with collected elements
        # 4. Pops loaded keys from sd in-place.
        # This is the behavior.
        # Keys remaining in sd are checked by the caller and warned "these keys remain", so be careful when recreating sd.

        return super().load_model_weights(sd, unet_prefix)  # self is returned
```

### 4. Model Interface (`comfy_intf.py`)

Bridges the actual model implementation with ComfyUI.

The constructor argument `operations` is important - it receives factory classes defined in `comfy.ops`.
Alternatively, when loaded with a custom loader like `GGUFLoader`, the corresponding class is passed.

This is a mechanism for hooking the processing of each `torch` module and performing VRAM management and decoding of quantized weights.

That is, when building a model, if you replace `torch` modules (e.g., `torch.nn.Linear`) with corresponding classes (e.g., `operations.Linear`), that module comes under ComfyUI's VRAM management and becomes a target for CPU offloading.
Or (as an example) when loaded with `GGUFLoader`, it will properly decode and process gguf's quantized weights at the necessary timing.
Support is not mandatory, but should be provided as much as possible.

```python
# models/comfy_intf.py

import torch
from torch import nn, Tensor
from .model import MyDiffusionModel  # Actual model implementation

class MyModelIntf(nn.Module):
    """Model interface for ComfyUI"""

    def __init__(self, *, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()

        # Operations can be processed in either way:
        # - Model implementation receives operations and uses them during model construction
        # - Model implementation uses torch.nn and modules are replaced later
        # Either is fine.
        # Below is the replacement approach.

        # Create model instance
        model = MyDiffusionModel(**kwargs)

        # Apply ComfyUI operations (quantization, etc.)
        if operations is not None:
            self._replace_modules(model, operations)

        self._dtype = dtype

        # Keep with property name corresponding to state_dict
        self.model = model.to(dtype=dtype, device=device)

    @property
    def dtype(self):
        # Caller expects this property to exist, so must define it
        # (Could be instance variable, but setter is never called, so defensively providing only getter here)
        return self._dtype

    @torch.inference_mode()
    def forward(self, x: Tensor, t: Tensor, transformer_options={}, **kwargs) -> Tensor:
        """
        Inference call from ComfyUI

        Args:
            x: Input latent (B, C, H, W)
            t: Timestep (B,)
            transformer_options: sigmas and callbacks
            kwargs: Additional inputs set in extra_conds

        Returns:
            Predicted noise or v-prediction, etc.
        """

        # kwargs = {
        #   context: Tensor,
        #   control, # ControlNet related
        #   # other key-value pairs returned by extra_conds
        # }

        # Get text embeddings, etc.
        cond = kwargs.pop('context')
        attention_mask = kwargs.pop('attention_mask', None)

        # Model inference
        output = self.model(t, x, cond, attention_mask)

        return output

    def _replace_modules(self, mod: nn.Module, operations):
        # Module replacement processing
        ...
```

### 5. Initialization and Hooks (`__init__.py`)

Create the entry point for custom nodes.

```python
# __init__.py

# Load custom nodes if available
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Apply model detection patch
from .utils.model_detection import apply_custom_detection_patch
apply_custom_detection_patch()

# Add to ComfyUI's supported model list
from comfy import supported_models
from .model.custom_model.supported_models import MyModel

if MyModel not in supported_models.models:
    supported_models.models.append(MyModel)
```

## Implementation Steps

### Step 1: Create Directory Structure

```
custom_nodes/
 └── my_model/
      ├── utils/
      |    └── model_detection.py
      ├── models/
      |    ├── supported_models.py
      |    ├── model_base.py
      |    ├── comfy_intf.py
      |    └── model.py
      └── __init__.py
```

### Step 2: Implement Model Detection

1. Find characteristic patterns to identify the model from state_dict keys
2. Extract model hyperparameters from state_dict
3. Add hook to `detect_unet_config`

### Step 3: Implement BaseModel

1. Inherit from `model_base.BaseModel`
2. Override `extra_conds` and `load_model_weights` as needed

### Step 4: Define Supported Model

1. Inherit from `supported_models_base.BASE`
2. Set `unet_config` for model identification
3. Configure model and text encoder settings

### Step 5: Implement Interface

1. Wrap torch model
2. Convert ComfyUI input format to your model's format

### Step 6: Initialize and Test

1. Execute hooks and model registration in `__init__.py`

### Miscellaneous

#### **Using Logs**
Follow ComfyUI's standard logging method:

```python
import logging
logger = logging.getLogger(__name__)
```

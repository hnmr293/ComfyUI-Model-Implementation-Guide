# ComfyUI ClipTarget Implementation Guide

## Overview

`ClipTarget` is a mechanism for integrating text encoders (CLIP, T5, Gemma, LLaMA, etc.) in ComfyUI. This guide provides detailed explanations on how to implement custom text encoders and integrate them with existing models.

## Role of ClipTarget

`ClipTarget` is a system that converts text prompts into embedding vectors, consisting of two components:

1. **Tokenizer**: Converts text to token IDs
2. **ClipModel**: Converts token IDs to embedding vectors

## Architecture

### System Configuration Diagram

```
User Prompt
    ↓
Tokenizer (SDTokenizer)
    ├── tokenize_with_weights()  [Text → Tokens + Weights]
    └── Textual Inversion Support
    ↓
ClipModel (SDClipModel)
    ├── encode_token_weights()  [Tokens → Embeddings]
    ├── Inference Execution
    └── Output Format Adjustment
    ↓
Conditioning Embeddings (context)
    ↓
Model
```

## Required Components for Implementation

### Example Directory Structure

```
custom_nodes/
 └── my_model/
      ├── models/
      |    ├── clip_comfy_intf.py
      |    └── clip_model.py  # <- Actual model implementation
      └── nodes.py
```

### 1. Tokenizer Implementation

The Tokenizer is implemented by inheriting from `sd1_clip.SDTokenizer`. Since `sd1_clip.SDTokenizer` internally calls `tokenizer_class.from_pretrained`, `transformers`' `PreTrainedTokenizer` can be used as is.

```python
# models/clip_comfy_intf.py

from comfy import sd1_clip
from transformers import CLIPTokenizer
import os

class MyTokenizer(sd1_clip.SDTokenizer):
    """Tokenizer using CLIP"""
    def __init__(
        self,
        tokenizer_path=None,
        max_length=77,
        pad_with_end=True,
        embedding_directory=None,
        embedding_size=768,
        embedding_key='clip_l',
        tokenizer_class=CLIPTokenizer,
        has_start_token=True,
        has_end_token=True,
        pad_to_max_length=True,
        min_length=None,
        pad_token=0,
        tokenizer_data={},
    ):
        # Having necessary files bundled reduces user burden
        if tokenizer_path is None:
            tokenizer_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "my_tokenizer"
            )

        super().__init__(
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            pad_with_end=pad_with_end,
            embedding_directory=embedding_directory,
            embedding_size=embedding_size,
            embedding_key=embedding_key,
            tokenizer_class=tokenizer_class,
            has_start_token=has_start_token,
            has_end_token=has_end_token,
            pad_to_max_length=pad_to_max_length,
            min_length=min_length,
            pad_token=pad_token,
            tokenizer_data=tokenizer_data,
        )
```

### 2. ClipModel Implementation

Wrapper class for actual model processing. Created by inheriting from `sd1_clip.SDClipModel`. Pass the torch model class to the `SDClipModel` constructor argument `model_class`. The generated model is set to `self.transformers`.

Either `load_sd` or `load_state_dict` is called depending on the node that initiated the loading. By default, `load_sd` calls `self.transformers.load_state_dict`. `load_state_dict` behaves according to `torch.nn.Module.load_state_dict`.

```python
# models/clip_comfy_intf.py

import torch
from comfy import sd1_clip
import comfy.ops
import json

# Model configuration (transformers config format)
MYCLIP_CONFIG = {
    'hidden_size': 768,
    'intermediate_size': 3072,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'vocab_size': 30522,
    'max_position_embeddings': 512,
    # Other settings...
}

# Special token definitions
MYCLIP_SPECIAL_TOKENS = {
    'start': 2,
    'end': 1,
    'pad': 0,
}

class MyClipModel(sd1_clip.SDClipModel):
    def __init__(
        self,
        device='cpu',
        layer='hidden',
        layer_idx=-2,  # Default CLIP_SKIP
        dtype=None,
        attention_mask=True,
        model_options={},
    ):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,  # So-called CLIP_SKIP. Overridden if CLIPSetLastLayer node is present
            textmodel_json_config=MYCLIP_CONFIG.copy(),
            dtype=dtype,
            special_tokens=MYCLIP_SPECIAL_TOKENS.copy(),
            model_class=MyModel,  # Model implementation class
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            return_projected_pooled=False,
            layer_norm_hidden_state=True,
            model_options=model_options,
        )

        # Caller expects this property to exist, so must define it
        self.dtypes = set([dtype])

    def load_sd(self, sd):
        """Called from CLIPLoader node"""
        # Perform key name conversion processing if necessary
        processed_sd = self._process_state_dict_keys(sd)
        return super().load_sd(processed_sd)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Called from LoadCheckpoint node"""
        # Example: Add transformer. prefix
        processed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('text_encoder.'):
                k = k[len('text_encoder.'):]
            processed_state_dict['transformer.' + k] = v

        return super().load_state_dict(processed_state_dict, strict, assign)

    def _process_state_dict_keys(self, sd):
        """Convert state_dict key names"""
        # Example
        processed = {}
        for k, v in sd.items():
            # Convert key names as needed
            # Example: model. -> transformer.
            if k.startswith('model.'):
                k = 'transformer.' + k[6:]
            processed[k] = v
        return processed
```

### 3. Model Implementation

Actual model implementation. For the constructor argument `operations`, please refer to the [Custom Model Implementation Guide](./Custom_Model_Guide.md).

```python
# models/clip_model.py

import torch
import torch.nn as nn
import comfy.ops

class MyModel(nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()

        # Build model using operations here
        ...

    def get_input_embeddings(self) -> nn.Embedding:
        ...

    def set_input_embeddings(self, embeddings: nn.Embedding):
        # Should not be called
        raise NotImplementedError

    def forward(
        self,
        input_ids,
        attention_mask=None,
        intermediate_output=None,
        final_layer_norm_intermediate=True,
        dtype=None
    ):
        ...
```

### 4. Creating Custom Node for Loading

The standard `CLIPLoader` node has hardcoded target types and doesn't allow external injection. It's best to create your own custom node.

> Note: It is technically possible by hooking `CLIPLoader.load_clip` itself.

For example, the implementation would be like this:

```python
# nodes.py

import re
import logging

import folder_paths
import comfy.utils
import comfy.sd1_clip
import comfy.sd
from comfy.supported_models_base import ClipTarget

from .models.clip_comfy_intf import MyTokenizer, MyClipModel


def _load_text_encoder_state_dicts(state_dicts=[], embedding_directory=None, model_options={}):
    clip_target = ClipTarget(MyTokenizer, MyClipModel)

    parameters = 0
    tokenizer_data = {}
    for sd in state_dicts:
        parameters += comfy.utils.calculate_parameters(sd)

    clip = comfy.sd.CLIP(clip_target, embedding_directory=embedding_directory, parameters=parameters, tokenizer_data=tokenizer_data, model_options=model_options)

    for sd in state_dicts:
        m, u = clip.load_sd(sd)
        if len(m) > 0:
            logging.warning(f'myclip missing: {m}')

        if len(u) > 0:
            logging.debug(f'myclip unexpected: {u}')

    return clip


class MyClipLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            'required': {
                'model_name': (folder_paths.get_filename_list('text_encoders'),),
            },
        }

    RETURN_TYPES = ('CLIP',)

    FUNCTION = 'load_clip'

    CATEGORY = 'my_nodes/loaders'

    def load_clip(self, model_name):
        model_path = folder_paths.get_full_path_or_raise('text_encoders', model_name)
        model_data = comfy.utils.load_torch_file(model_path, safe_load=True)

        model = _load_text_encoder_state_dicts([model_data])

        return (model,)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    'MyClipLoader': MyClipLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
#    'MyClipLoader': 'MyClipLoader',
}
```

## Implementation Steps

### Step 1: Create Directory Structure

```
custom_nodes/
 └── my_model/
      ├── models/
      |    ├── clip_comfy_intf.py
      |    └── clip_model.py
      ├── nodes.py
      └── __init__.py
```

### Step 2: Implement Tokenizer

1. Inherit from `sd1_clip.SDTokenizer`

### Step 3: Implement ClipModel

1. Inherit from `sd1_clip.SDClipModel`

### Step 4: Model Implementation

1. Wrap torch model

### Step 5: Create Custom Node for Loading

1. Create custom node
2. Register model in `__init__.py`

### Miscellaneous

#### **Using Logs**
Follow ComfyUI's standard logging method:

```python
import logging
logger = logging.getLogger(__name__)
```
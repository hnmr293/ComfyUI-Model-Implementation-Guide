# ComfyUI ClipTarget 実装ガイド

## 概要

`ClipTarget` は、ComfyUIでテキストエンコーダー（CLIP、T5、Gemma、LLaMAなど）を統合するための仕組みです。本ガイドでは、カスタムテキストエンコーダーの実装方法と、既存モデルへの統合方法を詳しく解説します。

## ClipTargetの役割

`ClipTarget` は、テキストプロンプトを埋め込みベクトルに変換するシステムで、以下の2つのコンポーネントから構成されます：

1. **Tokenizer**: テキストをトークンIDに変換
2. **ClipModel**: トークンIDを埋め込みベクトルに変換

## アーキテクチャ

### システム構成図

```
ユーザープロンプト
    ↓
Tokenizer (SDTokenizer)
    ├── tokenize_with_weights()  [テキスト→トークン+重み]
    └── Textual Inversion のサポート
    ↓
ClipModel (SDClipModel)
    ├── encode_token_weights()  [トークン→埋め込み]
    ├── 推論実行
    └── 出力形式の調整
    ↓
条件付け埋め込み（context）
    ↓
モデル
```

## 実装に必要なコンポーネント

### ディレクトリ構成例

```
custom_nodes/
 └── my_model/
      ├── models/
      |    ├── clip_comfy_intf.py
      |    └── clip_model.py  # <- 実際のモデル実装
      └── nodes.py
```

### 1. Tokenizer の実装

Tokenizerは `sd1_clip.SDTokenizer` を継承して実装します。`sd1_clip.SDTokenizer` は内部で `tokenizer_class.from_pretrained` を呼び出すので、`transformers` の `PreTrainedTokenizer` がそのまま使えます。

```python
# models/clip_comfy_intf.py

from comfy import sd1_clip
from transformers import CLIPTokenizer
import os

class MyTokenizer(sd1_clip.SDTokenizer):
    """CLIP を使用するトークナイザ"""
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
        # 必要なファイルは自分で持っておくとユーザーに負担がかからない
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

### 2. ClipModel の実装

実際のモデル処理を行うクラスのラッパーです。`sd1_clip.SDClipModel` を継承して作ります。`SDClipModel` のコンストラクタ引数 `model_class` に torch で作ったモデルのクラスを渡してやります。生成されたモデルは `self.transformers` に設定されます。

読み込みの起点となったノードによって `load_sd` か `load_state_dict` のどちらかが呼ばれます。デフォルトだと、`load_sd` は `self.transformers.load_state_dict` を呼び出します。`load_state_dict` は `torch.nn.Module.load_state_dict` の通り動作します。

```python
# models/clip_comfy_intf.py

import torch
from comfy import sd1_clip
import comfy.ops
import json

# モデル設定（transformers の config 形式）
MYCLIP_CONFIG = {
    'hidden_size': 768,
    'intermediate_size': 3072,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'vocab_size': 30522,
    'max_position_embeddings': 512,
    # その他の設定...
}

# 特殊トークンの定義
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
        layer_idx=-2,  # デフォルトの CLIP_SKIP
        dtype=None,
        attention_mask=True,
        model_options={},
    ):
        super().__init__(
            device=device,
            layer=layer,
            layer_idx=layer_idx,  # いわゆる CLIP_SKIP。CLIPSetLastLayerノードがあると上書きされる
            textmodel_json_config=MYCLIP_CONFIG.copy(),
            dtype=dtype,
            special_tokens=MYCLIP_SPECIAL_TOKENS.copy(),
            model_class=MyModel,  # モデル実装クラス
            enable_attention_masks=attention_mask,
            return_attention_masks=attention_mask,
            return_projected_pooled=False,
            layer_norm_hidden_state=True,
            model_options=model_options,
        )

        # 呼び出し側がこのプロパティの存在を期待しているので必ず定義する
        self.dtypes = set([dtype])

    def load_sd(self, sd):
        """CLIPLoader ノードから呼ばれた"""
        # 必要であればキー名の変換処理などを行う
        processed_sd = self._process_state_dict_keys(sd)
        return super().load_sd(processed_sd)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """LoadCheckpoint ノードから呼ばれた"""
        # 例：transformer.プレフィックスを追加
        processed_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('text_encoder.'):
                k = k[len('text_encoder.'):]
            processed_state_dict['transformer.' + k] = v

        return super().load_state_dict(processed_state_dict, strict, assign)

    def _process_state_dict_keys(self, sd):
        """state_dictのキー名を変換"""
        # 例
        processed = {}
        for k, v in sd.items():
            # 必要に応じてキー名を変換
            # 例: model. -> transformer.
            if k.startswith('model.'):
                k = 'transformer.' + k[6:]
            processed[k] = v
        return processed
```

### 3. モデル実装

実際のモデルの実装です。コンストラクタ引数の `operations` については [カスタムモデル実装ガイド](./Custom_Model_Guide_ja.md) を参照してください。

```python
# models/clip_model.py

import torch
import torch.nn as nn
import comfy.ops

class MyModel(nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        
        # ここで operations を使いながらモデルを構築する
        ...

    def get_input_embeddings(self) -> nn.Embedding:
        ...

    def set_input_embeddings(self, embeddings: nn.Embedding):
        # 呼ばれないはず
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

### 4. 読み込み用カスタムノードの作成

標準の `CLIPLoader` ノードでは対象のタイプがハードコードされており、外部からの注入ができません。自前でカスタムノードを作成するのがよいでしょう。

> いちおう、`CLIPLoader.load_clip` そのものをフックすれば可能です。

たとえば以下のような実装になります。

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

## 実装手順

### Step 1: ディレクトリ構造の作成

```
custom_nodes/
 └── my_model/
      ├── models/
      |    ├── clip_comfy_intf.py
      |    └── clip_model.py
      ├── nodes.py
      └── __init__.py
```

### Step 2: Tokenizer の実装

1. `sd1_clip.SDTokenizer`を継承

### Step 3: ClipModel の実装

1. `sd1_clip.SDClipModel`を継承

### Step 4: モデル実装

1. torch モデルをラップ

### Step 5: 読み込み用カスタムノードの作成

1. カスタムノードを作成
2. `__init__.py` でモデル登録

### その他

#### **ログの活用**
ロギングはComfyUI標準の方法にしたがうこと：

```python
import logging
logger = logging.getLogger(__name__)
```

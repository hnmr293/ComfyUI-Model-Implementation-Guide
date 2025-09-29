# ComfyUI カスタムモデル実装ガイド

## 概要

本ガイドでは、ComfyUIでカスタムモデルを実装する方法を解説します。既存のUNETLoaderやCheckpointLoaderSimpleから読み込み、ComfyUIのモデルシステムアーキテクチャと統合する方法を説明します。

## アーキテクチャ概要

### モデル読み込みフロー

```
UNETLoader / CheckpointLoaderSimple
    ↓
comfy.sd.load_diffusion_model / load_checkpoint_guess_config
    ↓
load_diffusion_model_state_dict
    ├── model_detection.detect_unet_config  [state_dictからモデル情報を抽出]
    ├── model_config_from_unet_config  [supported_modelsからマッチするモデルを検索]
    └── BASE.get_model  [BaseModelインスタンスを生成]
        ↓
    ModelPatcher  [最終的なモデルラッパー]
```

## 実装に必要なコンポーネント

### ディレクトリ構成例

```
custom_nodes/
 └── my_model/
      ├── utils/
      |    └── model_detection.py
      ├── models/
      |    ├── supported_models.py
      |    ├── model_base.py
      |    ├── comfy_intf.py
      |    └── model.py  # <- 実際のモデル実装がここにあるとする
      └── __init__.py
```

### 1. モデル検出システム (`model_detection.py`)

既存の `model_detection.detect_unet_config` では、重みファイルの中身（キー名）から対応するモデルの種類と設定値を判定しています。これから実装するモデルに対して適切に判定できるように、この処理をフックします。

`model_detection.detect_unet_config` の戻り値は辞書です。次節の内容の先取りになりますが、`supported_models.BASE` を継承したクラスのうち、プロパティがこの辞書の内容と一致するモデルが対応モデルとして選定されます。

```python
# utils/model_detection.py

import comfy.model_detection

_orig_detect_unet_config = comfy.model_detection.detect_unet_config

def apply_custom_detection_patch() -> None:
    """
    ComfyUIのモデル検出にフックを追加する。
    __init__.py などから呼び出してやることでカスタムモデルが検出されるようになる。
    なおここでは省略しているが、複数回呼び出されたときの対応もしておくべき。
    """
    comfy.model_detection.detect_unet_config = _detect_unet_config

def _detect_custom_model(state_dict, key_prefix):
    """カスタムモデルの検出ロジック"""
    # モデル特有のキーをチェック
    if f'{key_prefix}unique_layer.weight' not in state_dict:
        # 別のモデルだった
        return None

    # カスタムモデルの重みっぽいのでモデル設定を抽出する
    config = {
        'image_model': 'my_custom_model',
        'depth': len(key for key in state_dict if 'self_attn.to_q.weight' in key),
        'dim': state_dict[f'{key_prefix}embedder.weight'].size(0),
        'in_channels': 16,
        # その他の必要な設定...
    }

    return config

def _detect_unet_config(state_dict, key_prefix, *args, **kwargs):
    """detect_unet_config をフックする"""
    # 既存のワークフローを壊さないよう、先にオリジナル実装を呼び出すべき
    # ただし既存のモデルと構造が似ていて判定しきれない場合などは柔軟に対応してよいと思う
    unet_config = _orig_detect_unet_config(state_dict, key_prefix, *args, **kwargs)

    # 既存のモデルが見つからなければカスタムモデルの検出ロジックを呼び出す
    # 既存のモデルと似ており誤判定が起きる場合は `unet_config.get('image_model')` でモデル名を見て特殊処理を入れてもよい
    if unet_config is None:
        custom_config = _detect_custom_model(state_dict, key_prefix)
        if custom_config is not None:
            unet_config = custom_config

    return unet_config
```

### 2. BASE 定義 (`supported_models.py`)

ComfyUIのモデルシステムに統合するための設定クラスです。`model_detection.detect_unet_config` の戻り値の辞書と、このクラスのプロパティを比較してモデルが選定されます。プロパティはインスタンスから取得できればOKです。クラス変数、インスタンス変数のどちらでも構いません。以下のメソッドを定義する必要があります。
- `get_model`: 推論用のモデル (`BaseModel`) を返すようにします。
- `clip_target`: テキストエンコーダ (`ClipTarget`) を返すようにします。

```python
# models/supported_models.py

from comfy import supported_models_base, latent_formats
import torch

class MyModel(supported_models_base.BASE):
    """カスタムモデルのサポート定義"""

    # モデル識別用の設定
    unet_config = {
        'image_model': 'my_custom_model',
    }

    # 必須キー（state_dictに存在すべきキー）
    required_keys = []

    # メモリ使用量の係数
    # よくわからない (>_<)
    memory_usage_factor = 2.8

    # サポートする推論データ型
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    # latent の形式
    latent_format = latent_formats.Flux()

    def __init__(self, unet_config):
        super().__init__(unet_config)
        # unet_config の中身をもとに設定したければここで設定する
        #   self.latent_format = ...
        # など

    def get_model(self, state_dict, prefix='', device=None):
        """BaseModelインスタンスを生成"""
        return MyModelBase(self, device=device)

    @property
    def vae_key_prefix(self) -> list[str]:
        """
        VAE用のstate_dictプレフィックス（必要であれば）
        インスタンス変数やクラス変数にしてもよい
        """
        return ['autoencoder.']

    @property
    def text_encoder_key_prefix(self) -> list[str]:
        """
        テキストエンコーダ用のstate_dictプレフィックス（必要であれば）
        インスタンス変数やクラス変数にしてもよい
        """
        return ['text_encoder.']

    def clip_target(self, state_dict={}) -> supported_models_base.ClipTarget:
        """カスタムモデルが使用するCLIP/テキストエンコーダを返す"""
        # state_dictから判定してTokenizerとEncoderModelを返す
        from comfy.text_encoders import sd3_clip, flux
        prefix = self.text_encoder_key_prefix[0]
        t5_detect = sd3_clip.t5_xxl_detect(state_dict, f"{prefix}t5xxl.transformer.")
        return supported_models_base.ClipTarget(comfy.text_encoders.flux.FluxTokenizer, comfy.text_encoders.flux.flux_clip(**t5_detect))
```

### 3. BaseModelの実装 (`model_base.py`)

実際のモデル処理を行うクラスのラッパーです。`model_base.BaseModel` を継承して作ります。`BaseModel` のコンストラクタ引数 `unet_model` に torch で作ったモデルのクラスを渡してやります。生成されたモデルは `self.diffusion_model` に設定されます。

読み込みの起点となったノードによって `state_dict` のキー名と引数の `unet_prefix` が異なるので、必要に応じて適切に処理してください。

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
    カスタムモデルのラッパークラス。
    BaseModel.diffusion_model に実際のモデルが格納される。
    """

    def __init__(
        self,
        model_config: MyModel,         # 先ほど作成した BASE
        model_type=model_base.ModelType.EPS, # model_base.ModelType の列挙値から適切なものを選ぶ
        device=None,
    ):
        # unet_model パラメータに実装クラスを渡す
        super().__init__(
            model_config,
            model_type,
            device=device,
            unet_model=MyModelIntf,
        )

    def extra_conds(self, **kwargs) -> dict[str, Any]:
        """
        カスタムモデルの forward にキーワード引数で渡したいものを準備する。
        kwargs にはテキストエンコーダの出力が渡される。
        """
        out = super().extra_conds(**kwargs)

        # 例として attention_mask を追加してみる
        attention_mask = kwargs.get('attention_mask', None)
        if attention_mask is not None:
            out['attention_mask'] = comfy.conds.CONDRegular(attention_mask)

        return out

    def load_model_weights(self, sd: dict[str, Tensor], unet_prefix: str = '') -> "MyModelBase":
        """モデルの重みを読み込む"""
        # 必要に応じて sd のキー名を適切に処理すること。
        
        # sd のキー名および unet_prefix は読み込みの起点となったノードによって変わる
        # - CheckpointLoader から読み込まれた場合、unet_prefix は 'model.', 'net.', 'model.model.', 'model.diffusion_model.' のうち一つが自動で判定されて渡される。sd は読み込み時のまま。
        # - UNETLoader から読み込まれた場合、unet_prefix は空文字になっている。sd は自動判定されたプレフィックスが削除されて渡される。

        # BaseModel.load_model_weights は、
        # 1. 渡された sd から unet_prefix で始まるキーを集める（unet_prefix が空文字ならすべての要素が使われる）
        # 2. 集めたキーから unet_prefix を除去する
        # 3. 集めた要素で self.diffusion_model.load_state_dict を呼び出す
        # 4. 読み込まれたキーを sd から in-place に pop する。
        # という動作をする。
        # sd に残ったキーは呼び出し元で確認されて「このキーが残ってるよ」と警告を出すので、sd を作り直すときは注意。
        
        return super().load_model_weights(sd, unet_prefix)  # self が返される
```

### 4. モデルインターフェース (`comfy_intf.py`)

実際のモデル実装とComfyUIの橋渡しをします。

コンストラクタ引数 `operations` が重要で、`comfy.ops` で定義されたファクトリクラスが渡されます。
もしくは、`GGUFLoader` のようなカスタムローダーで読み込んだ場合は、対応するクラスが渡されます。

これは `torch` の各モジュールの処理をフックし、VRAM管理や量子化された重みのデコードを行うための仕組みです。

すなわち、モデル構築時に `torch` のモジュール（例：`torch.nn.Linear`）を対応するクラス（例：`operations.Linear`）で置き換えると、そのモジュールが ComfyUI の VRAM 管理下に入りCPUオフロードの対象となります。
あるいは（例として）`GGUFLoader` で読み込んだ場合、gguf の量子化された重みを必要なタイミングで適切にデコードして処理するようになります。
対応は必須ではないですが、できる限り対応すべきです。

```python
# models/comfy_intf.py

import torch
from torch import nn, Tensor
from .model import MyDiffusionModel  # 実際のモデル実装

class MyModelIntf(nn.Module):
    """ComfyUI用のモデルインターフェース"""

    def __init__(self, *, image_model=None, dtype=None, device=None, operations=None, **kwargs):
        super().__init__()

        # operations の処理方法は
        # - モデル実装側で operations を受け取ってモデル構築時に使用する
        # - モデル実装側は torch.nn を使用し、あとからモジュールを置き換える
        # のどちらでもよい。
        # 以下はあとから置き換える方式。

        # モデルインスタンスを作成
        model = MyDiffusionModel(**kwargs)

        # ComfyUIのoperationsを適用（量子化など）
        if operations is not None:
            self._replace_modules(model, operations)

        self._dtype = dtype
        
        # state_dict に対応するプロパティ名で持っておく
        self.model = model.to(dtype=dtype, device=device)

    @property
    def dtype(self):
        # 呼び出し側がこのプロパティの存在を期待しているので必ず定義する
        # （インスタンス変数でもいいが、setter が呼ばれることはないので、ここでは防御的に getter のみ提供しておく）
        return self._dtype

    @torch.inference_mode()
    def forward(self, x: Tensor, t: Tensor, transformer_options={}, **kwargs) -> Tensor:
        """
        ComfyUIからの推論呼び出し

        Args:
            x: 入力latent (B, C, H, W)
            t: タイムステップ (B,)
            transformer_options: sigmas や callback
            kwargs: extra_condsで設定した追加入力

        Returns:
            予測ノイズや v-prediction など
        """

        # kwargs = {
        #   context: Tensor,
        #   control, # ControlNet 関連
        #   # ほか extra_conds で返した key-value pairs
        # }

        # テキスト埋め込みなどを取得する
        cond = kwargs.pop('context')
        attention_mask = kwargs.pop('attention_mask', None)

        # モデル推論
        output = self.model(t, x, cond, attention_mask)

        return output
    
    def _replace_modules(self, mod: nn.Module, operations):
        # モジュールを置き換える処理
        ...
```

### 5. 初期化とフック (`__init__.py`)

カスタムノードのエントリーポイントを作成します。

```python
# __init__.py

# カスタムノードがあれば読み込み
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# モデル検出のパッチを適用
from .utils.model_detection import apply_custom_detection_patch
apply_custom_detection_patch()

# ComfyUIのサポートモデルリストに追加
from comfy import supported_models
from .model.custom_model.supported_models import MyModel

if MyModel not in supported_models.models:
    supported_models.models.append(MyModel)
```

## 実装手順

### Step 1: ディレクトリ構造の作成

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

### Step 2: モデル検出の実装

1. state_dictのキーからモデルを識別する特徴的なパターンを見つける
2. モデルのハイパーパラメータをstate_dictから抽出する
3. `detect_unet_config` にフックを追加

### Step 3: BaseModel の実装

1. `model_base.BaseModel` を継承
2. 必要に応じて `extra_conds`、`load_model_weights` をオーバーライド

### Step 4: サポートモデルの定義

1. `supported_models_base.BASE` を継承
2. モデル識別用の `unet_config` を設定
3. モデルとテキストエンコーダの設定

### Step 5: インターフェースの実装

1. torch モデルをラップ
2. ComfyUI の入力形式を自モデルの形式に変換

### Step 6: 初期化とテスト

1. `__init__.py` でフックとモデル登録を実行

### その他

#### **ログの活用**
ロギングはComfyUI標準の方法にしたがうこと：

```python
import logging
logger = logging.getLogger(__name__)
```

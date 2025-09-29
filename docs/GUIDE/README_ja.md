# ComfyUI カスタムモデル実装ガイド

このガイドは、ComfyUI 上で新しいモデルを定義・実装する手順を示したものです。

## 1. カスタムモデルの実装

[./Custom_Model_Guide_ja.md](Custom_Model_Guide_ja.md) を参照してください。
標準の CheckpointLoader などからカスタムモデルを読み込めるようにする手順を記載しています。

## 2. カスタムテキストエンコーダの実装

[./Custom_ClipTarget_Guide_ja.md](Custom_ClipTarget_Guide_ja.md) を参照してください。
カスタムテキストエンコーダを実装する手順を記載しています。

## 3. 補足

このガイドにしたがって実装を行う場合、全体のディレクトリ構成は以下のようになります。

```
custom_nodes/   # ComfyUI のカスタムノードディレクトリ
 └── my_model/  # このカスタムノードのルートディレクトリ
      ├── utils/
      |    └── model_detection.py   # モデル検出用フック
      ├── models/
      |    ├── supported_models.py  # モデル実装
      |    ├── model_base.py        # モデル実装
      |    ├── comfy_intf.py        # モデル実装
      |    ├── model.py             # モデル実装
      |    ├── clip_comfy_intf.py   # TE実装（必要な場合のみ）
      |    └── clip_model.py        # TE実装（必要な場合のみ）
      ├── nodes.py                  # TEのローダーノード（必要な場合のみ）
      └── __init__.py
```

# OAK-D Lite Playground

Raspberry Pi 5 + OAK-D Lite による顔認証・深度計測の実験リポジトリ。

## ハードウェア構成

- Raspberry Pi 5 (8GB)
- OAK-D Lite — USB 3.0（青ポート）に接続

## ソフトウェア環境

| | depthai 2.x（安定版） | depthai 3.x（開発中） |
|---|---|---|
| venv | `venv/` | `venv3/` |
| depthai | 2.27.0 | 3.5.0 |
| 主なスクリプト | `face_recognition_spatial.py` | `face_recognition_spatial_v3.py` / `face_recognition_auraface.py` |
| 状態 | ✅ 安定動作 | 🔧 動作確認済み・継続改善中 |

> **注意**: depthai 3.x は 2.x と API が大幅に異なるため、venv を分けて管理しています。

## セットアップ

### 1. USB 権限

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### 2. Pi 5 の USB 電力制限を解除

`/boot/firmware/config.txt` の `[all]` セクションに追記して再起動：

```
usb_max_current_enable=1
```

### 3. 仮想環境と依存パッケージ

```bash
# depthai 2.x（安定版）
python3 -m venv venv
source venv/bin/activate
pip install depthai==2.27.0 opencv-python Pillow blobconverter

# depthai 3.x（開発中）
python3 -m venv venv3
source venv3/bin/activate
pip install depthai opencv-python Pillow blobconverter onnxruntime huggingface_hub
```

### 4. モデルのダウンロード

```bash
# SFace（顔認証 - Apache 2.0）
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' \
  -o face_recognition_sface.onnx

# YuNet（顔アライメント用 - MIT）
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
  -o face_detection_yunet_2023mar.onnx

# AuraFace（顔認証 - Apache 2.0）
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='fal/AuraFace-v1', filename='glintr100.onnx', local_dir='.')
"
```

---

## スクリプト一覧

| スクリプト | venv | 説明 |
|---|---|---|
| `check_device.py` | venv | デバイス接続確認（カメラ一覧表示） |
| `capture.py` | venv | RGB + Depth のスナップショットを `/tmp/` に保存 |
| `live_view.py` | venv | RGB + Depth のリアルタイム表示 |
| `live_rgb.py` | venv | RGB のみのリアルタイム表示 |
| `face_recognition.py` | venv | 顔認証（手動深度取得版、depthai 2.x） |
| `face_recognition_spatial.py` | venv | 顔認証（depthai 2.x・**安定版**） |
| `face_recognition_spatial_v3.py` | venv3 | 顔認証 SFace（depthai 3.x 対応版） |
| `face_recognition_auraface.py` | venv3 | 顔認証 AuraFace（高精度・Apache 2.0） |

---

## パイプライン構成

### face_recognition_spatial_v3.py（SFace）

```mermaid
flowchart TB
    subgraph VPU["🔶 OAK-D Lite VPU (MyriadX)"]
        direction TB
        CAM_A["CAM_A\nRGB 640×400"]
        CAM_B["CAM_B\nNIR 640×400"]
        CAM_C["CAM_C\n右 640×400"]

        CAM_A -->|"BGR888p"| q_rgb["キュー: q_rgb\n表示用"]
        CAM_A -->|"BGR888p"| manip_rgb["ImageManip\n→ 300×300"]
        manip_rgb --> nn_rgb["NeuralNetwork\nface-detection-retail-0004\nRGB検出NN"]
        nn_rgb --> q_nn["キュー: q_nn"]

        CAM_B -->|"GRAY8"| q_mono["キュー: q_mono\n表示用"]
        CAM_B -->|"BGR888p"| manip_nir["ImageManip\n→ 300×300"]
        manip_nir --> nn_nir["NeuralNetwork\nface-detection-retail-0004\nNIR検出NN"]
        nn_nir --> q_nn_nir["キュー: q_nn_nir"]

        CAM_B -->|"left"| stereo["StereoDepth\n640×400\ndepthAlign=CAM_A"]
        CAM_C -->|"right"| stereo
        stereo --> q_depth["キュー: q_depth"]
    end

    subgraph Host["🔷 Raspberry Pi 5 ホスト"]
        direction TB
        switch{"m キー\nモード切替"}
        q_nn -->|"RGB モード"| switch
        q_nn_nir -->|"NIR モード"| switch

        switch --> parse["SSD 手動パース\ngetTensor('detection_out')\nreshape(-1, 7)"]
        parse --> track["host-side IoU トラッキング\ntrack_id 割当・TTL管理"]
        track --> yunet["YuNet\n顔ランドマーク検出\n→ alignCrop 112×112"]
        yunet --> sface["SFace\ncosine similarity\n→ DB マッチング"]
        sface --> display["表示\nBBox / ID / 認証結果\nsim:0.xx OK/NG"]

        q_depth --> depth["深度 ROI\nBBox中心 15×15px 中央値\n→ 距離 mm"]
        depth --> display
    end

    VPU -->|USB 3.0| Host
```

### face_recognition_auraface.py（AuraFace）

VPU パイプラインは SFace 版と同一。ホスト側の推論部分が異なる：

```mermaid
flowchart TB
    subgraph VPU["🔶 OAK-D Lite VPU (MyriadX)"]
        direction LR
        note["SFace 版と同一パイプライン\n（RGB/NIR 切替・StereoDepth）"]
    end

    subgraph Host["🔷 Raspberry Pi 5 ホスト"]
        direction TB

        subgraph main["メインスレッド（~10fps）"]
            det["顔検出 + IoU トラッキング"]
            det -->|"新規/更新 tracklet\n1.5秒以上経過"| enq["embed_queue.put_nowait()\nフルなら即スキップ"]
            det --> disp["表示ループ\nBBox / 認証結果キャッシュ表示"]
        end

        subgraph worker["埋め込みスレッド（~3fps）"]
            deq["embed_queue から取得"]
            deq --> pre["前処理\nBGR→RGB, 112×112\n(x-127.5)/128, NCHW"]
            pre --> ort["ONNX Runtime\nglintr100.onnx\nResNet100\n~330ms/枚"]
            ort --> l2["L2 正規化\n512次元埋め込み"]
            l2 --> match["cosine similarity\n→ DB マッチング"]
            match --> cache["track_results に書き込み\n(thread-safe)"]
        end

        cache -->|"最新結果を参照"| disp
    end

    VPU -->|USB 3.0| Host
```

### SFace vs AuraFace

| | SFace | AuraFace |
|---|---|---|
| モデル | SFace (ONNX) | GlintR100 / ResNet100 (ONNX) |
| モデルサイズ | 37MB | 261MB |
| 推論時間 (Pi 5 CPU) | ~20ms | ~330ms |
| 埋め込み次元 | 128 | 512 |
| ライセンス | Apache 2.0 | Apache 2.0 |
| デフォルト閾値 | sim > 0.65 | sim > 0.65 |

---

## 顔認証の使い方

```bash
# SFace (depthai 3.x)
source venv3/bin/activate
DISPLAY=:0 python3 face_recognition_spatial_v3.py

# AuraFace (depthai 3.x)
source venv3/bin/activate
DISPLAY=:0 python3 face_recognition_auraface.py

# SFace (depthai 2.x 安定版)
source venv/bin/activate
DISPLAY=:0 python3 face_recognition_spatial.py
```

| キー | 動作 |
|---|---|
| `r` | 顔を登録（名前を入力） |
| `m` | RGB ↔ NIR モード切り替え（検出 NN も切替） |
| `q` | 終了 |

```bash
# 登録データをリセット
rm face_db.pkl          # SFace用
rm face_db_auraface.pkl # AuraFace用
```

---

## チューニングパラメータ

```python
MIN_FACE_WIDTH       = 40     # 認証する最小顔幅 (px) ← 40で約2m前後まで対応
SIMILARITY_THRESHOLD = 0.65   # 同一人物判定ライン (0〜1、高いほど厳しい)
EMBED_INTERVAL       = 1.5    # AuraFace: 同一顔の再推論間隔 (秒)
DEPTH_MIN_MM         = 200    # 深度マップ表示の最小距離 (mm)
DEPTH_MAX_MM         = 5000   # 深度マップ表示の最大距離 (mm)
```

## 既知の制限・注意事項

- **depthai 3.x**: `SpatialDetectionNetwork` / `ObjectTracker` が OAK-D Lite で動作不安定 → ホスト側で代替実装
- **depthai 2.27.0 を推奨**: 3.x は ISP firmware クラッシュの回避策として Camera ノードを使用
- **NIR 登録推奨**: NIR モードで登録すると昼夜どちらでも認識精度が高い
- **OAK-D Lite は IR カメラ非搭載**: NIR は 940nm 外部照明 + モノカメラで代替

## モデル・ライセンス

| モデル | ライセンス | 商用利用 |
|---|---|---|
| face-detection-retail-0004 (Intel) | Apache 2.0 | ✅ |
| SFace | Apache 2.0 | ✅ |
| YuNet | MIT | ✅ |
| AuraFace (glintr100) | Apache 2.0 | ✅ |

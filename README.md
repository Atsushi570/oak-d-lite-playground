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
| 主なスクリプト | `face_recognition_spatial.py` | `face_recognition_spatial_v3.py` |
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
pip install depthai opencv-python Pillow blobconverter
```

### 4. モデルのダウンロード

```bash
# SFace（顔認証）
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' \
  -o face_recognition_sface.onnx

# YuNet（顔アライメント用ランドマーク検出）
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' \
  -o face_detection_yunet_2023mar.onnx
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
| `face_recognition_spatial_v3.py` | venv3 | 顔認証（depthai 3.x 対応版・**開発中**） |

---

## 顔認証（face_recognition_spatial.py）— 安定版

### パイプライン

```
OAK VPU:
  ColorCamera (CAM_A, RGB)
    └─ ImageManip (300x300)
         └─ MobileNetSpatialDetectionNetwork ─── 顔検出 + 距離取得
                └─ ObjectTracker ──────────────── 追尾 (track_id付与)
  MonoCamera (CAM_B) + MonoCamera (CAM_C)
    └─ StereoDepth ──────────────────────────── 深度マップ生成

Pi CPU:
  YuNet ─ 顔ランドマーク検出 (5点)
    └─ alignCrop() ─ 顔アライメント
         └─ SFace ─ 128次元埋め込み → コサイン類似度マッチング
```

### 使い方

```bash
source venv/bin/activate
DISPLAY=:0 python3 face_recognition_spatial.py
```

---

## 顔認証（face_recognition_spatial_v3.py）— depthai 3.x 対応版

### depthai 3.x 移行で変わった点

| 項目 | 2.x | 3.x |
|---|---|---|
| カメラノード | `ColorCamera` / `MonoCamera` | `Camera` + `requestOutput()` |
| XLinkOut | `XLinkOut` + `getOutputQueue()` | `output.createOutputQueue()` |
| 顔検出 | `MobileNetSpatialDetectionNetwork` | `NeuralNetwork` + ホスト側 SSD パース |
| 追尾 | `ObjectTracker` (VPU) | IoU ベースのホスト側トラッキング |
| 深度取得 | `spatialCoordinates.z` | 深度フレームから手動計算 |

> `MobileNetSpatialDetectionNetwork` と `ObjectTracker` は 3.x で動作問題があるため、  
> ホスト側で SSD 出力パースとトラッキングを実装しています。

### パイプライン

```
OAK VPU:
  Camera (CAM_A)
    ├─ requestOutput(640x400) → 表示用キュー
    └─ requestOutput(640x400) → ImageManip(300x300) → NeuralNetwork(face-detection)
  Camera (CAM_B) + Camera (CAM_C)
    └─ StereoDepth(DENSITY) → 深度マップ

Pi CPU:
  NeuralNetwork 生出力(1,1,200,7) → SSD 手動パース → IoU トラッキング
  YuNet + SFace → 顔認証
  深度フレーム BBox 中心 ROI median → 距離計算
```

### 使い方

```bash
source venv3/bin/activate
DISPLAY=:0 python3 face_recognition_spatial_v3.py
```

| キー | 動作 |
|---|---|
| `r` | 顔を登録（名前を入力） |
| `m` | RGB ↔ NIR モード切り替え |
| `q` | 終了 |

```bash
# 登録データをリセット
rm face_db.pkl
```

---

## チューニングパラメータ（共通）

```python
MIN_FACE_WIDTH       = 80     # 認証する最小顔幅 (px)
SIMILARITY_THRESHOLD = 0.65   # 同一人物判定ライン (0〜1、高いほど厳しい)
DEPTH_MIN_MM         = 200    # 深度マップ表示の最小距離 (mm)
DEPTH_MAX_MM         = 5000   # 深度マップ表示の最大距離 (mm)
```

## 精度向上のコツ

- 登録は **NIR モード**（`m` キーで切替）で行うと、昼夜どちらでも認識精度が高い
- 同じ人を複数アングル（正面・左・右）で登録すると角度変化への耐性が上がる
- 精度が低い場合は `SIMILARITY_THRESHOLD` を `0.55` 程度に下げて試す

## 既知の制限

- OAK-D Lite は IR カメラ非搭載。NIR 撮影は **940nm 外部照明 + モノカメラ** で代替
- SFace は RGB 訓練モデルのため、NIR モードでは精度がやや低下する場合がある
- depthai 3.x では `SpatialDetectionNetwork` の alignment エラーあり（OAK-D Lite 固有）
- ステレオ深度はテクスチャの少ない面（壁・天井）で不安定になる

---

## モデル・ライセンス

| モデル | ライセンス | 商用利用 |
|---|---|---|
| face-detection-retail-0004 (Intel) | Apache 2.0 | ✅ |
| SFace | Apache 2.0 | ✅ |
| YuNet | MIT | ✅ |

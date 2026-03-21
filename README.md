# OAK-D Lite Playground

Raspberry Pi 5 + OAK-D Lite を使った顔認証・深度計測の実験コード。

## 環境

- Raspberry Pi 5
- OAK-D Lite (USB 3.0接続)
- Python 3.11
- depthai 2.27.0

## セットアップ

```bash
python3 -m venv venv
source venv/bin/activate
pip install depthai==2.27.0 opencv-python Pillow blobconverter

# SFaceモデル
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx' -o face_recognition_sface.onnx

# YuNetモデル
curl -L 'https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx' -o face_detection_yunet_2023mar.onnx
```

## スクリプト

| スクリプト | 説明 |
|---|---|
| `check_device.py` | デバイス接続確認 |
| `capture.py` | RGB + Depth スナップショット保存 |
| `live_view.py` | RGB + Depth リアルタイム表示 |
| `live_rgb.py` | RGB のみリアルタイム表示 |
| `face_recognition.py` | 顔認証（手動深度取得版） |
| `face_recognition_spatial.py` | 顔認証（SpatialDetectionNetwork版・推奨） |

## 顔認証の使い方

```bash
source venv/bin/activate
DISPLAY=:0 python3 face_recognition_spatial.py

# 操作
#   r : 顔を登録（名前入力）
#   m : RGB ↔ NIR モード切り替え
#   q : 終了

# DB リセット
rm face_db.pkl
```

## 注意

- depthai 3.x は ISP firmware クラッシュあり → 2.27.0 を使う
- Pi 5 は usb_max_current_enable=1 が必要（/boot/firmware/config.txt）
- OAK-D Lite に IR カメラなし。NIR は 940nm 外部照明 + モノカメラで対応

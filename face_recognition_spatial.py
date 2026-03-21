"""
OAK-D Lite 顔認証（SFace + ObjectTracker + SpatialDetectionNetwork）
- OAK VPU: 顔検出 + ステレオ距離 + 追尾
- Pi CPU: SFace 埋め込み + コサイン類似度マッチング
- 正面・サイズフィルタ付き

操作:
  r  : カメラに向けて顔を登録（名前を入力）
  m  : カメラモード切替（RGB ↔ NIR）
  q  : 終了
"""
import depthai as dai
import numpy as np
import cv2
import warnings
import os
import pickle
import time
warnings.filterwarnings("ignore")

use_mono = False  # m キーでトグル

# ─── 設定 ─────────────────────────────────────────────
MIN_FACE_WIDTH   = 40      # 認証する最小顔幅 (px)
ASPECT_RATIO_MIN = 0.75    # 顔BBoxの縦横比 (正面は 0.9〜1.2 程度)
ASPECT_RATIO_MAX = 1.4
SIMILARITY_THRESHOLD = 0.65  # 類似度しきい値（大きい=類似）
DB_PATH = "face_db.pkl"
DEPTH_MIN_MM = 200         # 深度可視化の最小レンジ (mm)
DEPTH_MAX_MM = 5000        # 深度可視化の最大レンジ (mm)
# ──────────────────────────────────────────────────────

# SFace ロード
SFACE_MODEL = "face_recognition_sface.onnx"
recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")

# YuNet ロード（アライメント用）
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (112, 112),
                                          score_threshold=0.6,
                                          nms_threshold=0.3,
                                          top_k=1)

# 顔DBロード
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}  # {name: [embedding, ...]}

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

face_db = load_db()

def get_embedding(face_img):
    """顔クロップ → アライメント → 512次元埋め込み"""
    h, w = face_img.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(face_img)
    if faces is not None:
        aligned = recognizer.alignCrop(face_img, faces[0])
    else:
        aligned = cv2.resize(face_img, (112, 112))
    return recognizer.feature(aligned)

def match(emb):
    """最近傍マッチング → (name, similarity) or (None, -inf)"""
    best_name, best_sim = None, float("-inf")
    for name, embs in face_db.items():
        for db_emb in embs:
            score = recognizer.match(emb, db_emb, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_sim:
                best_sim = score
                best_name = name
    print(f"[MATCH] best={best_name} sim={best_sim:.3f} (threshold={SIMILARITY_THRESHOLD})", flush=True)
    return best_name, best_sim

def is_quality_face(x1, y1, x2, y2, frame_w, frame_h):
    """正面・サイズフィルタ"""
    w = x2 - x1
    h = y2 - y1
    if w < MIN_FACE_WIDTH:
        return False
    ratio = h / max(w, 1)
    if not (ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX):
        return False
    return True

# ─── depthai パイプライン ──────────────────────────────
pipeline = dai.Pipeline()

# ColorCamera (RGB)
cam_color = pipeline.createColorCamera()
cam_color.setPreviewSize(640, 400)
cam_color.setInterleaved(False)
cam_color.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_color.setFps(10)

# MonoCamera (NIR / CAM_B — left)
cam_mono = pipeline.createMonoCamera()
cam_mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_mono.setBoardSocket(dai.CameraBoardSocket.CAM_B)
cam_mono.setFps(10)

# MonoCamera (CAM_C — right)
cam_mono_r = pipeline.createMonoCamera()
cam_mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
cam_mono_r.setBoardSocket(dai.CameraBoardSocket.CAM_C)
cam_mono_r.setFps(10)

# StereoDepth
stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)

# フィルタ設定（face_recognition.py と同じ）
config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
stereo.initialConfig.set(config)

cam_mono.out.link(stereo.left)
cam_mono_r.out.link(stereo.right)

# 顔検出（SpatialDetectionNetwork — 検出 + 距離を VPU で取得）
# 300x300にリサイズ（face-detection-retail-0004 の入力サイズ）
manip = pipeline.createImageManip()
manip.initialConfig.setResize(300, 300)
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
cam_color.preview.link(manip.inputImage)


def _get_face_crop(frame, x1, y1, x2, y2):
    """顔クロップ（NIR検出時はBBoxがNIR座標なのでパディング不要）"""
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    return crop if crop.size > 0 else frame[0:1, 0:1]

blob_path = str(
    __import__('blobconverter').from_zoo(
        name="face-detection-retail-0004",
        shaves=6
    )
)

det_nn = pipeline.createMobileNetSpatialDetectionNetwork()
det_nn.setBlobPath(blob_path)
det_nn.setConfidenceThreshold(0.6)
det_nn.setBoundingBoxScaleFactor(0.5)   # depth取得のROIサイズ
det_nn.setDepthLowerThreshold(100)       # mm
det_nn.setDepthUpperThreshold(4000)      # mm
manip.out.link(det_nn.input)
stereo.depth.link(det_nn.inputDepth)

# ObjectTracker
tracker = pipeline.createObjectTracker()
tracker.setDetectionLabelsToTrack([1])
tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
det_nn.passthrough.link(tracker.inputTrackerFrame)
det_nn.passthrough.link(tracker.inputDetectionFrame)
det_nn.out.link(tracker.inputDetections)

# host へ
xout_rgb    = pipeline.createXLinkOut(); xout_rgb.setStreamName("rgb")
xout_mono   = pipeline.createXLinkOut(); xout_mono.setStreamName("mono")
xout_track  = pipeline.createXLinkOut(); xout_track.setStreamName("tracklets")
xout_depth  = pipeline.createXLinkOut(); xout_depth.setStreamName("depth")
cam_color.preview.link(xout_rgb.input)
cam_mono.out.link(xout_mono.input)
tracker.out.link(xout_track.input)
stereo.depth.link(xout_depth.input)

# ─── メインループ ──────────────────────────────────────
track_results = {}   # {track_id: (name, similarity, timestamp)}
last_depth_cache = {}  # {track_id: z_mm}
RESULT_TTL = 2.0     # 認証結果の表示保持時間(秒)

print("[RGB mode] (m で NIR に切替)")
print("起動中... (3秒待機)")
time.sleep(3)  # デバイス安定化待ち
print(f"登録済み: {list(face_db.keys())}")
print("r=登録  m=モード切替  q=終了")

with dai.Device(pipeline) as device:
    q_rgb    = device.getOutputQueue("rgb",       maxSize=4, blocking=False)
    q_mono   = device.getOutputQueue("mono",      maxSize=4, blocking=False)
    q_track  = device.getOutputQueue("tracklets", maxSize=4, blocking=False)
    q_depth  = device.getOutputQueue("depth",     maxSize=4, blocking=False)

    frame = None
    rgb_frame = None  # 認証・登録用（常にRGB）
    last_tracklets = None
    last_depth_frame = None
    while True:
        # RGB フレーム取得（常に取得してキューを消費）
        in_rgb = q_rgb.tryGet()
        # Mono フレーム取得（常に取得してキューを消費）
        in_mono = q_mono.tryGet()

        if use_mono:
            if in_mono:
                mono_cv = in_mono.getCvFrame()
                mono_resized = cv2.resize(mono_cv, (640, 400))
                frame = cv2.cvtColor(mono_resized, cv2.COLOR_GRAY2BGR)
        else:
            if in_rgb:
                frame = in_rgb.getCvFrame()
                rgb_frame = frame  # RGBを常時キープ

        # Depth フレーム取得（可視化用のみ）
        in_depth = q_depth.tryGet()
        if in_depth is not None:
            last_depth_frame = in_depth.getFrame().astype(np.float32)

        in_track = q_track.tryGet()
        if in_track is not None:
            last_tracklets = in_track
        if in_track and frame is not None:
            h, w = frame.shape[:2]
            display = frame.copy()

            for t in in_track.tracklets:
                if t.status.name in ("LOST", "REMOVED"):
                    last_depth_cache.pop(t.id, None)
                    continue

                roi = t.roi.denormalize(w, h)
                x1, y1 = int(roi.topLeft().x), int(roi.topLeft().y)
                x2, y2 = int(roi.bottomRight().x), int(roi.bottomRight().y)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                tid = t.id

                quality = is_quality_face(x1, y1, x2, y2, w, h)
                color = (0, 200, 0) if quality else (100, 100, 100)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID:{tid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # 質フィルタ通過 → 認証
                if quality:
                    face_crop = _get_face_crop(frame, x1, y1, x2, y2)
                    if face_crop.size > 0:
                        emb = get_embedding(face_crop)
                        if face_db:
                            name, sim = match(emb)
                            if sim > SIMILARITY_THRESHOLD:
                                track_results[tid] = (name, sim, time.time())
                            else:
                                track_results[tid] = ("Unknown", sim, time.time())
                        else:
                            track_results[tid] = ("(no DB)", 0, time.time())

                # 距離表示（SpatialDetectionNetwork の spatialCoordinates を使用）
                z_mm = int(t.spatialCoordinates.z)
                if z_mm > 0:
                    last_depth_cache[tid] = z_mm  # 有効値をキャッシュ
                else:
                    z_mm = last_depth_cache.get(tid, 0)  # キャッシュから取得

                if z_mm > 0:
                    dist_label = f"{z_mm // 1000}.{(z_mm % 1000) // 100}m"
                    cv2.putText(display, dist_label, (x2 - 50, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)

                # 認証結果表示
                now = time.time()
                if tid in track_results:
                    name, sim, ts = track_results[tid]
                    if now - ts < RESULT_TTL:
                        if name == "(no DB)":
                            label = "[no DB]"
                            lcolor = (0, 80, 255)
                        elif name == "Unknown":
                            label = f"Unknown  sim:{sim:.2f} NG"
                            lcolor = (0, 80, 255)
                        else:
                            label = f"{name}  sim:{sim:.2f} OK"
                            lcolor = (0, 255, 0)
                        cv2.putText(display, label, (x1, y2+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lcolor, 2)

            # 凡例（左上に常時表示）
            cv2.putText(display, "sim: similarity (0=diff, 1=same)",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(display, f"threshold: {SIMILARITY_THRESHOLD}",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # モード表示（右上）
            mode_label = "[NIR]" if use_mono else "[RGB]"
            (tw, th), _ = cv2.getTextSize(mode_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(display, mode_label, (w - tw - 10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Recognition (SFace + Spatial)", display)

        # Depth 可視化ウィンドウ
        if last_depth_frame is not None:
            d = last_depth_frame
            d_vis = np.clip((DEPTH_MAX_MM - d) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255, 0, 255)
            d_vis[d == 0] = 0  # 無効ピクセルは黒
            colored = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)
            # 各顔の BBox を白い矩形で描画
            if last_tracklets is not None:
                dh, dw = colored.shape[:2]
                for t in last_tracklets.tracklets:
                    if t.status.name in ("LOST", "REMOVED"):
                        continue
                    roi = t.roi.denormalize(dw, dh)
                    dx1, dy1 = max(0, int(roi.topLeft().x)), max(0, int(roi.topLeft().y))
                    dx2, dy2 = min(dw, int(roi.bottomRight().x)), min(dh, int(roi.bottomRight().y))
                    cv2.rectangle(colored, (dx1, dy1), (dx2, dy2), (255, 255, 255), 1)
            cv2.imshow("Depth", colored)

        key = cv2.waitKey(1) & 0xFF

        # m: モード切替
        if key == ord('m'):
            use_mono = not use_mono
            mode_str = "NIR" if use_mono else "RGB"
            print(f"[モード切替] → {mode_str}")

        # r: 登録モード
        elif key == ord('r') and frame is not None:
            name = input("登録する名前: ").strip()
            if name:
                # 現在フレームの最大顔を登録
                best = None
                best_area = 0
                if last_tracklets:
                    for t in last_tracklets.tracklets:
                        if t.status.name in ("LOST", "REMOVED"):
                            continue
                        roi = t.roi.denormalize(w, h)
                        x1, y1 = max(0,int(roi.topLeft().x)), max(0,int(roi.topLeft().y))
                        x2, y2 = min(w,int(roi.bottomRight().x)), min(h,int(roi.bottomRight().y))
                        area = (x2-x1)*(y2-y1)
                        if area > best_area:
                            best_area = area
                            best = frame[y1:y2, x1:x2]
                if best is not None and best.size > 0:
                    emb = get_embedding(best)
                    if name not in face_db:
                        face_db[name] = []
                    face_db[name].append(emb)
                    save_db(face_db)
                    print(f"登録完了: {name} (合計 {len(face_db[name])} 枚)")
                else:
                    print("顔が見つかりませんでした")

        elif key == ord('q'):
            break

cv2.destroyAllWindows()
print("終了")

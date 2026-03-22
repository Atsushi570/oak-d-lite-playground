"""
OAK-D Lite 顔認証 - depthai 3.x 対応版
- OAK VPU: NeuralNetwork(顔検出) + StereoDepth
- Pi CPU: SSD出力手動パース + IoTトラッキング + SFace認証

操作:
  r  : 顔を登録（名前を入力）
  m  : RGB <-> NIR モード切替
  q  : 終了
"""
import depthai as dai
import numpy as np

# ─── ヘッドポーズ推定用 3D 顔モデル（mm単位、鼻先を原点）────────────────────
FACE_3D_MODEL = np.array([
    [ 0.0,   0.0,   0.0 ],   # 鼻先
    [-30.0, -28.0, -30.0],   # 左目
    [ 30.0, -28.0, -30.0],   # 右目
    [-22.0,  25.0, -30.0],   # 左口角
    [ 22.0,  25.0, -30.0],   # 右口角
], dtype=np.float64)
import cv2
import warnings
import os
import pickle
import time
import argparse
import onnxruntime as ort
warnings.filterwarnings("ignore")

# ─── 引数 ──────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--antispoof", action="store_true",
                help="MiniFASNetV2 による RGB anti-spoof を有効化（デフォルト: 無効）")
ap.add_argument("--name", default=None, help="起動時に登録する名前")
args = ap.parse_args()
USE_ANTISPOOF = args.antispoof

use_mono = False

# ─── 設定 ─────────────────────────────────────────────
MIN_FACE_WIDTH       = 40
ASPECT_RATIO_MIN     = 0.75
ASPECT_RATIO_MAX     = 1.4
SIMILARITY_THRESHOLD = 0.65
CONFIDENCE_THRESHOLD = 0.6
DB_PATH              = "face_db.pkl"
DEPTH_MIN_MM         = 200
DEPTH_MAX_MM         = 5000
RESULT_TTL           = 2.0
TRACK_IOU_THRESH     = 0.3
TRACK_TTL            = 1.0

# ─── host-side tracking ───────────────────────────────
host_tracks     = {}   # {tid: {'bbox': (x1,y1,x2,y2), 'last_seen': float}}
rgb_host_tracks = {}   # depth チェック用: 常に RGB(CAM_A) 座標の BBox を管理
next_track_id = [0]

def calc_iou(a, b):
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (ua + 1e-6)

# ─── SFace / YuNet ────────────────────────────────────
recognizer   = cv2.FaceRecognizerSF.create("face_recognition_sface.onnx", "")
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (112, 112),
    score_threshold=0.6, nms_threshold=0.3, top_k=1)

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

face_db = load_db()

def get_embedding(face_img):
    h, w = face_img.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(face_img)
    if faces is not None:
        aligned = recognizer.alignCrop(face_img, faces[0])
    else:
        aligned = cv2.resize(face_img, (112, 112))
    return recognizer.feature(aligned)

def match(emb):
    best_name, best_sim = None, float("-inf")
    for name, embs in face_db.items():
        for db_emb in embs:
            score = recognizer.match(emb, db_emb, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_sim:
                best_sim, best_name = score, name
    return best_name, best_sim

def is_quality_face(x1, y1, x2, y2, fw, fh):
    w = x2 - x1
    h = y2 - y1
    if w < MIN_FACE_WIDTH:
        return False
    ratio = h / max(w, 1)
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX


def _get_face_crop(frame, x1, y1, x2, y2):
    """顔クロップ（NIR検出時はBBoxがNIR座標なのでパディング不要）"""
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    return crop if crop.size > 0 else frame[0:1, 0:1]

def estimate_pose(frame, x1, y1, x2, y2):
    """YuNet + solvePnP でヘッドポーズ（Yaw/Pitch/Roll）を推定"""
    crop = _get_face_crop(frame, x1, y1, x2, y2)
    ch, cw = crop.shape[:2]
    if ch < 20 or cw < 20:
        return None

    face_detector.setInputSize((cw, ch))
    _, faces = face_detector.detect(crop)
    if faces is None or len(faces) == 0:
        return None

    lm = faces[0][4:14].reshape(5, 2)  # 右目/左目/鼻/右口/左口 (crop座標)

    # 2D: solvePnP の対応点（3Dモデルと同順: 鼻/左目/右目/左口/右口）
    image_pts = np.array([
        lm[2], lm[1], lm[0], lm[4], lm[3]
    ], dtype=np.float64)

    fh, fw = frame.shape[:2]
    focal = float(fw)
    cam_mat = np.array([
        [focal, 0,     fw / 2.0],
        [0,     focal, fh / 2.0],
        [0,     0,     1.0     ],
    ], dtype=np.float64)

    # crop→frame 座標へオフセット（表示用ランドマーク）
    ox, oy = max(0, x1), max(0, y1)
    lm_frame = (lm + np.array([ox, oy])).astype(int)

    # solvePnP（crop 座標系で解く → 画角は同一なので cam_mat はフレーム基準でOK）
    image_pts_frame = image_pts + np.array([ox, oy])
    ok, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, image_pts_frame, cam_mat,
        np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
    if not ok:
        return None

    # 回転行列 → Euler 角
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = np.degrees(np.arctan2( rmat[1, 0], rmat[0, 0]))
        roll  = np.degrees(np.arctan2( rmat[2, 1], rmat[2, 2]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw   = 0.0
        roll  = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))

    # 3軸を投影（origin=鼻先, X=赤, Y=緑, Z=青）
    axis_3d = np.float32([[0,0,0],[50,0,0],[0,-50,0],[0,0,-50]])
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, cam_mat, np.zeros((4,1)))
    axis_2d = axis_2d.reshape(-1, 2).astype(int)

    return {
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "axis_2d": axis_2d,
        "landmarks": lm_frame,
    }


def check_liveness_depth(depth_frame, x1, y1, x2, y2, border=30):
    """顔/背景 深度差分チェック。常に (result_or_None, stats) を返す"""
    if depth_frame is None:
        return None, {"face": 0, "bg": 0, "diff": 0, "std": 0.0, "fpx": 0}
    dh, dw = depth_frame.shape[:2]
    fx1, fy1 = max(0, x1), max(0, y1)
    fx2, fy2 = min(dw, x2), min(dh, y2)
    face_roi = depth_frame[fy1:fy2, fx1:fx2]
    total_px = max(1, (fx2 - fx1) * (fy2 - fy1))
    face_valid = face_roi[(face_roi > 100) & (face_roi < 4000)]
    fpx = face_valid.size
    valid_ratio = fpx / total_px
    # 有効ピクセルが30%未満 = 反射面・近すぎ → SPOOF（取れない=怪しい）
    if valid_ratio < 0.3:
        face_depth = float(np.median(face_valid)) if fpx > 0 else 0
        face_std = float(np.std(face_valid)) if fpx > 1 else 0.0
        return "SPOOF", {"face": int(face_depth), "bg": 0, "diff": 0, "std": round(face_std, 1), "fpx": fpx, "ratio": round(valid_ratio, 2)}
    face_depth = float(np.median(face_valid))
    face_std = float(np.std(face_valid))
    # 0.4m 未満はステレオ信頼性低 → SPOOF
    if face_depth < 400:
        return "SPOOF", {"face": int(face_depth), "bg": 0, "diff": 0, "std": round(face_std, 1), "fpx": fpx, "ratio": round(valid_ratio, 2)}
    bx1, by1 = max(0, x1 - border), max(0, y1 - border)
    bx2, by2 = min(dw, x2 + border), min(dh, y2 + border)
    bg_full = depth_frame[by1:by2, bx1:bx2].copy()
    bg_full[fy1 - by1:fy2 - by1, fx1 - bx1:fx2 - bx1] = 0
    bg_valid = bg_full[(bg_full > 100) & (bg_full < 4000)]
    if bg_valid.size < 10:
        return None, {"face": int(face_depth), "bg": 0, "diff": 0, "std": round(face_std, 1), "fpx": fpx}
    bg_depth = float(np.median(bg_valid))
    diff = bg_depth - face_depth
    # 実顔: std=27-33mm / スマホ(平面): std=0.6-1.4mm / 背景ノイズ混入: std=300mm+
    # 上限100mmで背景混入を除外、下限15mmで平面を除外
    result = "LIVE" if 15.0 < face_std < 100.0 else "SPOOF"
    stats = {"face": int(face_depth), "bg": int(bg_depth), "diff": int(diff), "std": round(face_std, 1), "fpx": fpx, "ratio": round(valid_ratio, 2)}
    return result, stats


# ─── Anti-spoof (onnxruntime, host CPU) ──────────────
ANTISPOOF_MODEL = "anti_spoof_mn3.onnx"
_spoof_sess = None

def get_spoof_session():
    global _spoof_sess
    if _spoof_sess is None and os.path.exists(ANTISPOOF_MODEL):
        _spoof_sess = ort.InferenceSession(ANTISPOOF_MODEL, providers=['CPUExecutionProvider'])
    return _spoof_sess

def check_liveness_nn(face_crop):
    """MiniFASNetV2 (80x80, 3クラス) で顔クロップの liveness 判定"""
    sess = get_spoof_session()
    if sess is None:
        return None
    try:
        resized = cv2.resize(face_crop, (80, 80))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = rgb.transpose(2, 0, 1)[np.newaxis]  # NCHW
        input_name = sess.get_inputs()[0].name
        logits = sess.run(None, {input_name: inp})[0].flatten()
        # softmax → MiniFASNetV2: class 2=real/live
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        real_prob = float(probs[2])
        print(f"[ANTISPOOF] real={real_prob:.3f} probs={probs}", flush=True)
        return "LIVE" if real_prob > 0.5 else "SPOOF"
    except Exception as e:
        print(f"[ANTISPOOF] error: {e}", flush=True)
        return None

# ─── Blob ─────────────────────────────────────────────
import blobconverter
blob_path = str(blobconverter.from_zoo(
    name="face-detection-retail-0004", shaves=6))

# ─── 状態変数 ──────────────────────────────────────────
track_results      = {}
last_depth_cache   = {}
nn_result_cache    = {}   # {tid: "LIVE"/"SPOOF"/None}
last_nn_ts         = {}   # {tid: timestamp} rate limit 用

print("[RGB mode] (m で NIR に切替)")
print("起動中... (3秒待機)")
time.sleep(3)
print(f"登録済み: {list(face_db.keys())}")
print("r=登録  m=モード切替  q=終了")

with dai.Device() as device:
    p = dai.Pipeline(device)

    # ─── カメラ ───────────────────────────────────────
    cam_color = p.create(dai.node.Camera)
    cam_color.build(dai.CameraBoardSocket.CAM_A)
    rgb_disp = cam_color.requestOutput((640, 400), type=dai.ImgFrame.Type.BGR888p, fps=10)
    rgb_nn   = cam_color.requestOutput((640, 400), type=dai.ImgFrame.Type.BGR888p, fps=10)

    cam_mono_l = p.create(dai.node.Camera)
    cam_mono_l.build(dai.CameraBoardSocket.CAM_B)
    mono_out_l = cam_mono_l.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=10)
    mono_out_nir_nn = cam_mono_l.requestOutput((640, 400), type=dai.ImgFrame.Type.BGR888p, fps=10)  # NIR detection用（RGBと同じ座標空間）

    cam_mono_r = p.create(dai.node.Camera)
    cam_mono_r.build(dai.CameraBoardSocket.CAM_C)
    mono_out_r = cam_mono_r.requestOutput((640, 400), type=dai.ImgFrame.Type.GRAY8, fps=10)

    # ─── StereoDepth ──────────────────────────────────
    stereo = p.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(640, 400)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(True)  # MinZ を ~40cm → ~20cm に改善
    stereo.initialConfig.postProcessing.speckleFilter.enable = True
    stereo.initialConfig.postProcessing.speckleFilter.speckleRange = 28
    stereo.initialConfig.postProcessing.temporalFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.enable = True
    stereo.initialConfig.postProcessing.spatialFilter.holeFillingRadius = 2
    mono_out_l.link(stereo.left)
    mono_out_r.link(stereo.right)

    # ─── ImageManip → NeuralNetwork ───────────────────
    manip = p.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(300, 300)
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    rgb_nn.link(manip.inputImage)

    det_nn = p.create(dai.node.NeuralNetwork)
    det_nn.setBlobPath(blob_path)
    manip.out.link(det_nn.input)

    # NIR 用検出NN
    manip_nir = p.create(dai.node.ImageManip)
    manip_nir.initialConfig.setOutputSize(300, 300)
    manip_nir.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    mono_out_nir_nn.link(manip_nir.inputImage)
    det_nn_nir = p.create(dai.node.NeuralNetwork)
    det_nn_nir.setBlobPath(blob_path)
    manip_nir.out.link(det_nn_nir.input)

    # ─── キュー ───────────────────────────────────────
    q_rgb   = rgb_disp.createOutputQueue(maxSize=4, blocking=False)
    q_mono  = mono_out_l.createOutputQueue(maxSize=4, blocking=False)
    q_nn     = det_nn.out.createOutputQueue(maxSize=4, blocking=False)
    q_nn_nir = det_nn_nir.out.createOutputQueue(maxSize=4, blocking=False)
    q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    p.start()

    frame           = None
    last_depth_frame = None

    while True:
        # フレーム取得
        in_rgb  = q_rgb.tryGet()
        in_mono = q_mono.tryGet()

        if use_mono:
            if in_mono:
                mono_cv = in_mono.getCvFrame()
                frame = cv2.cvtColor(cv2.resize(mono_cv, (640, 400)), cv2.COLOR_GRAY2BGR)
        else:
            if in_rgb:
                frame = in_rgb.getCvFrame()
                rgb_frame = frame  # RGBを常時キープ

        # 深度取得
        in_depth = q_depth.tryGet()
        if in_depth is not None:
            last_depth_frame = in_depth.getFrame().astype(np.float32)

        # NN出力: 常に両方読む（RGB=depth用、NIR=表示用）
        in_nn_rgb = q_nn.tryGet()
        in_nn_nir = q_nn_nir.tryGet()
        in_nn = in_nn_nir if use_mono else in_nn_rgb

        def parse_dets(in_nn_data, w, h):
            if in_nn_data is None:
                return []
            dets = []
            raw = np.array(in_nn_data.getTensor('detection_out')).reshape(-1, 7)
            for det in raw:
                if float(det[2]) < CONFIDENCE_THRESHOLD:
                    continue
                x1 = max(0, int(det[3] * w))
                y1 = max(0, int(det[4] * h))
                x2 = min(w, int(det[5] * w))
                y2 = min(h, int(det[6] * h))
                if x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2))
            return dets

        current_dets = []
        rgb_current_dets = []
        if frame is not None:
            h, w = frame.shape[:2]
            current_dets = parse_dets(in_nn, w, h)
            rgb_current_dets = parse_dets(in_nn_rgb, w, h)

        # host-side tracking
        now = time.time()
        for bbox in current_dets:
            best_tid, best_iou = None, TRACK_IOU_THRESH
            for tid, info in host_tracks.items():
                iou = calc_iou(bbox, info['bbox'])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is None:
                best_tid = next_track_id[0]
                next_track_id[0] += 1
            host_tracks[best_tid] = {'bbox': bbox, 'last_seen': now}
            matched_dets[best_tid] = bbox

        # 古いトラックを削除
        for tid in list(host_tracks.keys()):
            if now - host_tracks[tid]['last_seen'] > TRACK_TTL:
                del host_tracks[tid]
                last_depth_cache.pop(tid, None)
                track_results.pop(tid, None)
                nn_result_cache.pop(tid, None)
                last_nn_ts.pop(tid, None)

        # RGB tracks 更新（depth チェック用・常に CAM_A 座標）
        for bbox in rgb_current_dets:
            best_tid, best_iou = None, TRACK_IOU_THRESH
            for tid, info in rgb_host_tracks.items():
                iou = calc_iou(bbox, info['bbox'])
                if iou > best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is None:
                # 表示トラックと位置で対応付ける
                for tid, info in host_tracks.items():
                    iou = calc_iou(bbox, info['bbox'])
                    if iou > TRACK_IOU_THRESH and tid not in rgb_host_tracks:
                        best_tid = tid
                        break
                if best_tid is None:
                    best_tid = f"rgb_{id(bbox)}"
            rgb_host_tracks[best_tid] = {'bbox': bbox, 'last_seen': now}
        for tid in list(rgb_host_tracks.keys()):
            if now - rgb_host_tracks[tid]['last_seen'] > TRACK_TTL:
                del rgb_host_tracks[tid]

        # 表示用: 常に host_tracks 全体から再構築（NN フレームが来ないときも BBox を維持）
        matched_dets = {tid: info['bbox'] for tid, info in host_tracks.items()}

        # 表示
        if frame is not None:
            display = frame.copy()
            fh, fw = display.shape[:2]

            for tid, (x1, y1, x2, y2) in matched_dets.items():
                quality = is_quality_face(x1, y1, x2, y2, fw, fh)
                # depth は CAM_A(RGB) 座標 → RGB BBox を優先使用
                rgb_bbox = rgb_host_tracks.get(tid, {}).get('bbox', None)
                if rgb_bbox is None:
                    # IoU で近い rgb track を探す
                    best_rgb_bbox, best_iou = None, 0.3
                    for rinfo in rgb_host_tracks.values():
                        iou = calc_iou((x1,y1,x2,y2), rinfo['bbox'])
                        if iou > best_iou:
                            best_iou, best_rgb_bbox = iou, rinfo['bbox']
                    rgb_bbox = best_rgb_bbox or (x1, y1, x2, y2)
                dx1, dy1, dx2, dy2 = rgb_bbox
                depth_live, depth_stats = check_liveness_depth(last_depth_frame, dx1, dy1, dx2, dy2)
                # per-track NN 推論 (--antispoof 指定時のみ)
                nn_result = None
                if USE_ANTISPOOF:
                    if quality and (now - last_nn_ts.get(tid, 0)) > 1.5:
                        face_crop_spoof = _get_face_crop(frame, x1, y1, x2, y2)
                        if face_crop_spoof.size > 0:
                            nn_result_cache[tid] = check_liveness_nn(face_crop_spoof)
                            last_nn_ts[tid] = now
                    nn_result = nn_result_cache.get(tid)
                # 判定ロジック
                if USE_ANTISPOOF:
                    # マルチモーダル: depth=None は判定保留（LIVE と断言しない）
                    if nn_result == "SPOOF" or depth_live == "SPOOF":
                        liveness = "SPOOF"
                    elif nn_result == "LIVE" and depth_live == "LIVE":
                        liveness = "LIVE"
                    else:
                        liveness = None
                else:
                    # 深度のみ
                    liveness = depth_live  # "LIVE" / "SPOOF" / None
                if liveness == "LIVE":
                    color = (0, 200, 0)
                elif liveness == "SPOOF":
                    color = (0, 0, 255)
                else:
                    color = (0, 200, 200)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, f"ID:{tid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if liveness:
                    lbl_detail = f"{liveness} nn:{nn_result or '?'} d:{depth_live or '?'}"
                    cv2.putText(display, lbl_detail, (x1, y1-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                # 深度stats を BBox 内下部に常に重畳（None でも表示）
                if depth_stats is not None:
                    s = depth_stats
                    # depth_liveがNoneの場合は赤みがかった色で警告表示
                    stat_color = (200, 200, 0) if depth_live is not None else (80, 80, 255)
                    ratio_pct = int(s.get('ratio', 0) * 100)
                    txt1 = f"std:{s['std']}mm diff:{s['diff']:+d}mm ratio:{ratio_pct}%"
                    txt2 = f"face:{s['face']}mm px:{s['fpx']}"
                    cv2.putText(display, txt1, (x1+2, y2-18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, stat_color, 1)
                    cv2.putText(display, txt2, (x1+2, y2-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, stat_color, 1)

                # ヘッドポーズ推定
                pose = estimate_pose(frame, x1, y1, x2, y2)
                if pose:
                    ax = pose["axis_2d"]
                    org = tuple(ax[0])
                    cv2.line(display, org, tuple(ax[1]), (0,   0, 255), 2)  # X: 赤
                    cv2.line(display, org, tuple(ax[2]), (0, 255,   0), 2)  # Y: 緑
                    cv2.line(display, org, tuple(ax[3]), (255,  0,   0), 2)  # Z: 青
                    cv2.putText(display,
                                f"Y:{pose['yaw']:+.0f} P:{pose['pitch']:+.0f} R:{pose['roll']:+.0f}",
                                (x1, y2+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

                if quality and liveness != "SPOOF":
                    face_crop = _get_face_crop(frame, x1, y1, x2, y2)
                    if face_crop.size > 0:
                        emb = get_embedding(face_crop)
                        if face_db:
                            name, sim = match(emb)
                            if sim > SIMILARITY_THRESHOLD:
                                track_results[tid] = (name, sim, now)
                            else:
                                track_results[tid] = ("Unknown", sim, now)
                        else:
                            track_results[tid] = ("(no DB)", 0, now)

                # 距離（depth frame から手動計算）
                z_mm = 0
                if last_depth_frame is not None:
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    r = 15
                    dh, dw = last_depth_frame.shape[:2]
                    roi = last_depth_frame[max(0,cy-r):min(dh,cy+r), max(0,cx-r):min(dw,cx+r)]
                    valid = roi[(roi > 100) & (roi < 4000)]
                    if valid.size > 0:
                        z_mm = int(np.median(valid))
                if z_mm > 0:
                    last_depth_cache[tid] = z_mm
                else:
                    z_mm = last_depth_cache.get(tid, 0)
                if z_mm > 0:
                    cv2.putText(display, f"{z_mm//1000}.{(z_mm%1000)//100}m",
                                (x2-50, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                # 認証結果
                if tid in track_results:
                    rname, rsim, rts = track_results[tid]
                    if now - rts < RESULT_TTL:
                        if rname == "(no DB)":
                            lbl, lclr = "[no DB]", (0, 80, 255)
                        elif rname == "Unknown":
                            lbl, lclr = f"Unknown sim:{rsim:.2f}", (0, 80, 255)
                        else:
                            lbl, lclr = f"{rname} sim:{rsim:.2f} OK", (0, 255, 0)
                        cv2.putText(display, lbl, (x1, y2+36),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lclr, 2)

            # 凡例・モード
            cv2.putText(display, f"sim thresh:{SIMILARITY_THRESHOLD}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            mode_lbl = "[NIR]" if use_mono else "[RGB]"
            (tw, _), _ = cv2.getTextSize(mode_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(display, mode_lbl, (fw-tw-10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Face Recognition (SFace)", display)

        # Depth 可視化
        if last_depth_frame is not None:
            d = last_depth_frame
            invalid_mask = (d == 0) | (d >= 65535) | (d > DEPTH_MAX_MM)
            d_vis = np.clip((DEPTH_MAX_MM - d) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255, 0, 255)
            colored = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)
            colored[invalid_mask] = 0
            for tid, (x1, y1, x2, y2) in matched_dets.items():
                cv2.rectangle(colored, (x1, y1), (x2, y2), (255, 255, 255), 1)
            # カラーバー凡例
            ch = colored.shape[0]
            bar_w, label_w = 30, 60
            bar_vals = np.linspace(255, 0, ch).astype(np.uint8).reshape(ch, 1)
            bar_gray = np.repeat(bar_vals, bar_w, axis=1)
            bar_color = cv2.applyColorMap(bar_gray, cv2.COLORMAP_JET)
            label_area = np.zeros((ch, label_w, 3), dtype=np.uint8)
            labels = [(f"{DEPTH_MIN_MM/1000:.1f}m", 0.05), ("1m", 1000), ("2m", 2000),
                      ("3m", 3000), ("4m", 4000), (f"{DEPTH_MAX_MM/1000:.0f}m", 0.95)]
            for txt, pos in labels:
                if isinstance(pos, float):
                    y = int(pos * ch)
                else:
                    y = int((pos - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * ch)
                y = max(12, min(ch - 4, y))
                cv2.putText(label_area, txt, (2, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.imshow("Depth", np.hstack([colored, bar_color, label_area]))

        # 顔がいないときはスリープを長くして CPU 節約
        wait_ms = 1 if matched_dets else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('m'):
            use_mono = not use_mono
            print(f"モード: {'NIR' if use_mono else 'RGB'}")

        elif key == ord('r') and frame is not None:
            name = input("登録する名前: ").strip()
            if name and matched_dets:
                # 最大面積の顔を登録
                best_bbox = max(matched_dets.values(), key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x1, y1, x2, y2 = best_bbox
                face_crop = _get_face_crop(frame, x1, y1, x2, y2)
                if face_crop.size > 0:
                    emb = get_embedding(face_crop)
                    face_db.setdefault(name, []).append(emb)
                    save_db(face_db)
                    print(f"登録完了: {name} (合計 {len(face_db[name])} 枚)")
                else:
                    print("顔が見つかりませんでした")
            elif name:
                print("顔が検出されていません")

        elif key == ord('q'):
            break

cv2.destroyAllWindows()
print("終了")

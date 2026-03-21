"""
OAK-D Lite 顔認証 - AuraFace (glintr100.onnx) 版
- OAK VPU: NeuralNetwork(顔検出) + StereoDepth
- Pi CPU: SSD 手動パース + IoU トラッキング
- Background Thread: AuraFace 推論（約 330ms/frame → 非同期処理）

AuraFace (fal/AuraFace-v1): Apache 2.0 ライセンス / 商用利用可

操作:
  r  : 顔を登録（名前を入力）
  m  : RGB <-> NIR モード切替
  q  : 終了
"""
import depthai as dai
import numpy as np
import cv2
import warnings
import os
import pickle
import time
import threading
import queue
import onnxruntime as ort
warnings.filterwarnings("ignore")

use_mono = False

# ─── 設定 ─────────────────────────────────────────────
MIN_FACE_WIDTH       = 40
ASPECT_RATIO_MIN     = 0.75
ASPECT_RATIO_MAX     = 1.4
SIMILARITY_THRESHOLD = 0.65   # cosine similarity（SFaceと同じ基準、高いほど類似）
CONFIDENCE_THRESHOLD = 0.6
DB_PATH              = "face_db_auraface.pkl"
DEPTH_MIN_MM         = 200
DEPTH_MAX_MM         = 5000
RESULT_TTL           = 3.0
TRACK_IOU_THRESH     = 0.3
TRACK_TTL            = 1.0
EMBED_INTERVAL       = 1.5    # 同一 track_id の再推論間隔 (秒)

# ─── host-side tracking ───────────────────────────────
host_tracks    = {}   # {tid: {'bbox': (x1,y1,x2,y2), 'last_seen': float}}
next_track_id  = [0]

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

# ─── AuraFace (ONNX Runtime) ──────────────────────────
AURAFACE_MODEL = "glintr100.onnx"
aura_sess = ort.InferenceSession(AURAFACE_MODEL, providers=['CPUExecutionProvider'])
aura_input_name = aura_sess.get_inputs()[0].name

# YuNet（アライメント用）
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx", "", (112, 112),
    score_threshold=0.6, nms_threshold=0.3, top_k=1)

def preprocess_auraface(face_img):
    """112x112 に揃えて NCHW RGB 正規化"""
    img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) / 128.0
    return img.transpose(2, 0, 1)[np.newaxis]  # (1,3,112,112)

def align_face(face_img):
    """YuNet でランドマーク検出してアライメント"""
    h, w = face_img.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(face_img)
    if faces is not None and len(faces) > 0:
        # 5点ランドマーク → 112x112 にアライン
        # YuNet の faces[0] から手動クロップ（alignCrop は SFace 専用のため）
        x, y, fw, fh = [int(v) for v in faces[0][:4]]
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x+fw), min(h, y+fh)
        cropped = face_img[y:y2, x:x2]
        if cropped.size > 0:
            return cv2.resize(cropped, (112, 112))
    return cv2.resize(face_img, (112, 112))

def get_embedding(face_img):
    """顔クロップ → 512 次元 L2 正規化済み埋め込み"""
    aligned = align_face(face_img)
    inp = preprocess_auraface(aligned)
    out = aura_sess.run(None, {aura_input_name: inp})[0][0]  # (512,)
    norm = np.linalg.norm(out)
    return out / (norm + 1e-6)

def cosine_distance(a, b):
    """コサイン距離（0=同一, 2=正反対）"""
    return 1.0 - float(np.dot(a, b))

def match(emb):
    """最近傍マッチング → (name, similarity)"""
    best_name, best_sim = None, float("-inf")
    for name, embs in face_db.items():
        for db_emb in embs:
            sim = 1.0 - cosine_distance(emb, db_emb)  # similarity: 高いほど類似
            if sim > best_sim:
                best_sim, best_name = sim, name
    print(f"[MATCH] best={best_name} sim={best_sim:.3f} (threshold={SIMILARITY_THRESHOLD})", flush=True)
    return best_name, best_sim

def is_quality_face(x1, y1, x2, y2, fw, fh):
    w = x2 - x1
    h = y2 - y1
    if w < MIN_FACE_WIDTH:
        return False
    ratio = h / max(w, 1)
    return ASPECT_RATIO_MIN <= ratio <= ASPECT_RATIO_MAX

# ─── 顔DB ──────────────────────────────────────────────
def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_db(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

face_db = load_db()

# ─── バックグラウンド埋め込みスレッド ──────────────────
embed_queue   = queue.Queue(maxsize=2)   # (tid, face_crop, is_register, register_name)
track_results = {}   # {tid: (name, sim, timestamp)}
results_lock  = threading.Lock()
last_embed_ts = {}   # {tid: float} 最後に推論した時刻

def embedding_worker():
    while True:
        item = embed_queue.get()
        if item is None:
            break  # 終了シグナル
        tid, face_crop, is_register, reg_name = item
        try:
            emb = get_embedding(face_crop)
            if is_register:
                # 登録モード
                if reg_name not in face_db:
                    face_db[reg_name] = []
                face_db[reg_name].append(emb)
                save_db(face_db)
                print(f"登録完了: {reg_name} (合計 {len(face_db[reg_name])} 枚)", flush=True)
            else:
                # 認証モード
                if face_db:
                    name, sim = match(emb)
                    result = (name, sim, time.time())
                else:
                    result = ("(no DB)", 0.0, time.time())
                with results_lock:
                    track_results[tid] = result
        except Exception as e:
            print(f"埋め込みエラー: {e}", flush=True)
        finally:
            embed_queue.task_done()

# スレッド起動
embed_thread = threading.Thread(target=embedding_worker, daemon=True)
embed_thread.start()


def _get_face_crop(frame, x1, y1, x2, y2):
    """顔クロップ（NIR検出時はBBoxがNIR座標なのでパディング不要）"""
    h, w = frame.shape[:2]
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    return crop if crop.size > 0 else frame[0:1, 0:1]

# ─── Blob ─────────────────────────────────────────────
blob_path = str(__import__('blobconverter').from_zoo(
    name="face-detection-retail-0004", shaves=6))

# ─── 状態変数 ──────────────────────────────────────────
last_depth_cache = {}

print("[AuraFace mode] RGB (m で NIR に切替)")
print("起動中... (3秒待機)")
time.sleep(3)
print(f"登録済み: {list(face_db.keys())}")
print(f"推論速度目安: ~330ms/frame (バックグラウンドスレッド処理)")
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

    frame            = None
    last_depth_frame = None
    matched_dets     = {}

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

        # NN 出力: SSD 手動パース
        in_nn = (q_nn_nir if use_mono else q_nn).tryGet()
        current_dets = []
        if in_nn is not None and frame is not None:
            h, w = frame.shape[:2]
            raw = np.array(in_nn.getTensor('detection_out')).reshape(-1, 7)
            for det in raw:
                conf = float(det[2])
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                x1 = max(0, int(det[3] * w))
                y1 = max(0, int(det[4] * h))
                x2 = min(w, int(det[5] * w))
                y2 = min(h, int(det[6] * h))
                if x2 > x1 and y2 > y1:
                    current_dets.append((x1, y1, x2, y2))

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

        # 古いトラックを削除
        for tid in list(host_tracks.keys()):
            if now - host_tracks[tid]['last_seen'] > TRACK_TTL:
                del host_tracks[tid]
                last_depth_cache.pop(tid, None)
                last_embed_ts.pop(tid, None)
                with results_lock:
                    track_results.pop(tid, None)

        matched_dets = {tid: info['bbox'] for tid, info in host_tracks.items()}

        # 品質チェック通過した顔をバックグラウンドキューへ（rate limit あり）
        if frame is not None:
            fh, fw = frame.shape[:2]
            for tid, (x1, y1, x2, y2) in matched_dets.items():
                if not is_quality_face(x1, y1, x2, y2, fw, fh):
                    continue
                last_ts = last_embed_ts.get(tid, 0)
                if now - last_ts < EMBED_INTERVAL:
                    continue  # まだ interval 内
                face_crop = _get_face_crop(frame, x1, y1, x2, y2).copy()
                if face_crop.size == 0:
                    continue
                try:
                    embed_queue.put_nowait((tid, face_crop, False, None))
                    last_embed_ts[tid] = now
                except queue.Full:
                    pass  # キュー満杯ならスキップ

        # 表示
        if frame is not None:
            display = frame.copy()
            fh, fw = display.shape[:2]

            for tid, (x1, y1, x2, y2) in matched_dets.items():
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(display, f"ID:{tid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

                # 距離
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

                # 認証結果（スレッドからのキャッシュ）
                with results_lock:
                    result = track_results.get(tid)
                if result:
                    rname, rsim, rts = result
                    if now - rts < RESULT_TTL:
                        if rname == "(no DB)":
                            lbl, lclr = "[no DB]", (0, 80, 255)
                        elif rsim > SIMILARITY_THRESHOLD:
                            lbl, lclr = f"{rname} sim:{rsim:.2f} OK", (0, 255, 0)
                        else:
                            lbl, lclr = f"Unknown sim:{rsim:.2f}", (0, 80, 255)
                        cv2.putText(display, lbl, (x1, y2+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, lclr, 2)
                else:
                    # 推論中表示
                    if last_embed_ts.get(tid, 0) > 0:
                        cv2.putText(display, "analyzing...", (x1, y2+18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

            # 凡例
            cv2.putText(display, f"AuraFace sim>{SIMILARITY_THRESHOLD}=OK",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
            mode_lbl = "[NIR]" if use_mono else "[RGB]"
            (tw, _), _ = cv2.getTextSize(mode_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(display, mode_lbl, (fw-tw-10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Face Recognition (AuraFace)", display)

        # Depth 可視化
        if last_depth_frame is not None:
            d = last_depth_frame
            d_vis = np.clip((DEPTH_MAX_MM - d) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255, 0, 255)
            d_vis[(d == 0) | (d >= 65535) | (d > DEPTH_MAX_MM)] = 0
            colored = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)
            for tid, (x1, y1, x2, y2) in matched_dets.items():
                cv2.rectangle(colored, (x1, y1), (x2, y2), (255, 255, 255), 1)
            cv2.imshow("Depth", colored)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('m'):
            use_mono = not use_mono
            print(f"モード: {'NIR' if use_mono else 'RGB'}")

        elif key == ord('r') and frame is not None and matched_dets:
            name = input("登録する名前: ").strip()
            if name:
                best_bbox = max(matched_dets.values(), key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x1, y1, x2, y2 = best_bbox
                face_crop = _get_face_crop(frame, x1, y1, x2, y2).copy()
                if face_crop.size > 0:
                    print("登録中... (AuraFace 推論 ~330ms)")
                    try:
                        embed_queue.put((best_bbox[0], face_crop, True, name), timeout=5)
                    except queue.Full:
                        pass

        elif key == ord('q'):
            break

# 終了
embed_queue.put(None)
embed_thread.join(timeout=2)
cv2.destroyAllWindows()
print("終了")

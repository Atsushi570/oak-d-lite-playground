import depthai as dai
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore")

print(f"depthai {dai.__version__}")
print("RGB + Depth ライブビュー起動中... (終了: q)")

# 深度の表示範囲（固定）: 200mm〜5000mm
DEPTH_MIN_MM = 200
DEPTH_MAX_MM = 5000

pipeline = dai.Pipeline()

# RGB
camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 400)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(20)
xoutRgb = pipeline.createXLinkOut()
xoutRgb.setStreamName("rgb")
camRgb.preview.link(xoutRgb.input)

# ステレオ深度
monoL = pipeline.createMonoCamera()
monoR = pipeline.createMonoCamera()
monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoL.setFps(20)
monoR.setFps(20)

stereo = pipeline.createStereoDepth()
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)              # サブピクセル精度向上
# depthAlign不要（ColorCamera未接続）

# 空間フィルタ（ノイズ低減）
config = stereo.initialConfig.get()
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.temporalFilter.enable = True   # 時間方向スムージング
config.postProcessing.spatialFilter.enable = True    # 空間スムージング
config.postProcessing.spatialFilter.holeFillingRadius = 2
stereo.initialConfig.set(config)

monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xoutDepth = pipeline.createXLinkOut()
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)

with dai.Device(pipeline) as device:
    qRgb   = device.getOutputQueue("rgb",   maxSize=4, blocking=False)
    qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

    while True:
        inRgb = qRgb.tryGet()
        if inRgb is not None:
            cv2.imshow("RGB", inRgb.getCvFrame())

        inDepth = qDepth.tryGet()
        if inDepth is not None:
            d = inDepth.getFrame().astype(np.float32)
            # 固定レンジで正規化（近い=赤、遠い=青）
            d_vis = np.clip((DEPTH_MAX_MM - d) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255, 0, 255)
            d_vis[d == 0] = 0  # 無効ピクセルは黒
            colored = cv2.applyColorMap(d_vis.astype(np.uint8), cv2.COLORMAP_JET)
            # 中央の距離
            h, w = d.shape
            c = int(d[h//2, w//2])
            if c > 0:
                cv2.putText(colored, f"{c}mm ({c/1000:.2f}m)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow("Depth", colored)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

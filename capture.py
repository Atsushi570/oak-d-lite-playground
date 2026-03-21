import depthai as dai
import numpy as np
import time
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

def capture_rgb():
    print("[RGB] 接続中...")
    with dai.Device() as device:
        p = dai.Pipeline(device)
        cam = p.create(dai.node.ColorCamera)
        cam.setIspScale(1, 3)
        q = cam.isp.createOutputQueue()
        p.start()
        time.sleep(2.0)
        for _ in range(30):
            f = q.tryGet()
            if f is not None:
                return f.getCvFrame()
            time.sleep(0.1)
    return None

def capture_depth():
    print("[Depth] 接続中...")
    with dai.Device() as device:
        p = dai.Pipeline(device)
        monoL = p.create(dai.node.MonoCamera)
        monoR = p.create(dai.node.MonoCamera)
        monoL.setCamera("left")
        monoR.setCamera("right")
        monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        stereo = p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DENSITY)
        stereo.setLeftRightCheck(True)
        monoL.out.link(stereo.left)
        monoR.out.link(stereo.right)
        q = stereo.depth.createOutputQueue()
        p.start()
        time.sleep(2.0)
        for _ in range(30):
            f = q.tryGet()
            if f is not None:
                return f.getCvFrame()
            time.sleep(0.1)
    return None

# RGB
rgb = capture_rgb()
if rgb is not None:
    print(f"RGB OK: {rgb.shape}")
    Image.fromarray(rgb[:, :, ::-1]).save('/tmp/oak_rgb.jpg', quality=90)
    print("→ /tmp/oak_rgb.jpg")
else:
    print("RGB NG")

time.sleep(1.0)

# Depth
depth = capture_depth()
if depth is not None:
    print(f"Depth OK: {depth.shape}")
    d = depth.astype(np.float32)
    valid = d[d > 0]
    d_max = np.percentile(valid, 95) if len(valid) else 1.0
    d_norm = np.clip(d / d_max * 255, 0, 255).astype(np.uint8)
    Image.fromarray(d_norm).save('/tmp/oak_depth.jpg', quality=90)
    print("→ /tmp/oak_depth.jpg")
    h, w = depth.shape
    c = int(depth[h//2, w//2])
    print(f"中央距離: {c} mm ({c/1000:.2f} m)")
else:
    print("Depth NG")

print("\n完了 ✅")

import depthai as dai
import cv2

print(f"depthai {dai.__version__}")
print("RGB ライブビュー起動中... (終了: q)")

pipeline = dai.Pipeline()

camRgb = pipeline.createColorCamera()
camRgb.setPreviewSize(640, 360)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(20)

xout = pipeline.createXLinkOut()
xout.setStreamName("rgb")
camRgb.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    frame_count = 0
    while True:
        f = q.get()
        if f is not None:
            frame = f.getCvFrame()
            frame_count += 1
            if frame_count == 1:
                print(f"フレーム取得OK: {frame.shape}")
            cv2.imshow("RGB Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

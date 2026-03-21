import depthai as dai
import sys

print(f'depthai version: {dai.__version__}')

available = dai.Device.getAllAvailableDevices()
print(f'検出されたデバイス数: {len(available)}')

if not available:
    print('ERROR: デバイスが見つかりません')
    sys.exit(1)

for dev_info in available:
    print(f'  - MxID: {dev_info.getMxId()}  state: {dev_info.state}')

print('\nデバイスに接続中...')
with dai.Device() as device:
    print(f'接続成功: {device.getDeviceName()}')
    cameras = device.getConnectedCameras()
    print(f'接続カメラ: {cameras}')
    print('\n✅ 動作確認完了!')

import cv2
import numpy as np
from ctypes import *
from MvImport.MvCameraControl_class import *
from MvImport.PixelType_header import *

class MVSController:
    def __init__(self):
        self.cam = None
        self.data_buf = None
        self.nPayloadSize = None

    def enum_device(self, tlayerType, deviceList):
        """
        枚举设备
        :param tlayerType: 枚举传输层
        :param deviceList: 设备列表
        """
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("enum devices fail! ret[0x%x]" % ret)
            sys.exit()

        if deviceList.nDeviceNum == 0:
            print("find no device!")
            sys.exit()

        print("Find %d devices!" % deviceList.nDeviceNum)

        for i in range(0, deviceList.nDeviceNum):
            mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                strModeName = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName)
                print("device model name: %s" % strModeName)
                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
            elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
                print("\nu3v device: [%d]" % i)
                strModeName = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName if per != 0)
                print("device model name: %s" % strModeName)
                strSerialNumber = "".join(chr(per) for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber if per != 0)
                print("user serial number: %s" % strSerialNumber)

    def is_image_color(self, enType):
        dates = {
            PixelType_Gvsp_RGB8_Packed: 'color',
            PixelType_Gvsp_BGR8_Packed: 'color',
            PixelType_Gvsp_YUV422_Packed: 'color',
            PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
            PixelType_Gvsp_BayerGR8: 'color',
            PixelType_Gvsp_BayerRG8: 'color',
            PixelType_Gvsp_BayerGB8: 'color',
            PixelType_Gvsp_BayerBG8: 'color',
            PixelType_Gvsp_BayerGB10: 'color',
            PixelType_Gvsp_BayerGB10_Packed: 'color',
            PixelType_Gvsp_BayerBG10: 'color',
            PixelType_Gvsp_BayerBG10_Packed: 'color',
            PixelType_Gvsp_BayerRG10: 'color',
            PixelType_Gvsp_BayerRG10_Packed: 'color',
            PixelType_Gvsp_BayerGR10: 'color',
            PixelType_Gvsp_BayerGR10_Packed: 'color',
            PixelType_Gvsp_BayerGB12: 'color',
            PixelType_Gvsp_BayerGB12_Packed: 'color',
            PixelType_Gvsp_BayerBG12: 'color',
            PixelType_Gvsp_BayerBG12_Packed: 'color',
            PixelType_Gvsp_BayerRG12: 'color',
            PixelType_Gvsp_BayerRG12_Packed: 'color',
            PixelType_Gvsp_BayerGR12: 'color',
            PixelType_Gvsp_BayerGR12_Packed: 'color',
            PixelType_Gvsp_Mono8: 'mono',
            PixelType_Gvsp_Mono10: 'mono',
            PixelType_Gvsp_Mono10_Packed: 'mono',
            PixelType_Gvsp_Mono12: 'mono',
            PixelType_Gvsp_Mono12_Packed: 'mono'}
        return dates.get(enType, '未知')

    def enable_device(self, nConnectionNum, deviceList):
        """
        设备使能
        :param nConnectionNum: 设备编号
        :return: 相机, 图像缓存区, 图像数据大小
        """
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("create handle fail! ret[0x%x]" % ret)
            sys.exit()

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("open device fail! ret[0x%x]" % ret)
            sys.exit()

        if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
            nPacketSize = self.cam.MV_CC_GetOptimalPacketSize()
            if int(nPacketSize) > 0:
                ret = self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nPacketSize)
                if ret != 0:
                    print("Warning: Set Packet Size fail! ret[0x%x]" % ret)
            else:
                print("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("set trigger mode fail! ret[0x%x]" % ret)
            sys.exit()

        stParam = MVCC_INTVALUE()  # 使用 CameraParams_header.py 中定义的结构体
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            print("get payload size fail! ret[0x%x]" % ret)
            sys.exit()

        self.nPayloadSize = stParam.nCurValue
        self.data_buf = (c_ubyte * self.nPayloadSize)()
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("start grabbing fail! ret[0x%x]" % ret)
            sys.exit()

        return self.cam, self.data_buf, self.nPayloadSize
    def get_image(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))
        
        ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
        if ret != 0:
            print("get one frame fail! ret[0x%x]" % ret)
            return None
        
        # 使用memmove安全复制数据
        pData = (c_ubyte * stOutFrame.stFrameInfo.nFrameLen)()
        memmove(byref(pData), stOutFrame.pBufAddr, stOutFrame.stFrameInfo.nFrameLen)
        data = np.frombuffer(pData, count=int(stOutFrame.stFrameInfo.nFrameLen), dtype=np.uint8)
        
        enType = stOutFrame.stFrameInfo.enPixelType
        if self.is_image_color(enType) == 'color':
            if enType in [PixelType_Gvsp_BayerGR8, PixelType_Gvsp_BayerRG8, PixelType_Gvsp_BayerGB8, PixelType_Gvsp_BayerBG8]:
                img = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
                img = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
            else:
                img = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth, 3))
        else:
            img = data.reshape((stOutFrame.stFrameInfo.nHeight, stOutFrame.stFrameInfo.nWidth))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # 释放图像缓存
        self.cam.MV_CC_FreeImageBuffer(stOutFrame)
        return img
    
    def close_device(self):
        """
        关闭设备
        """
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("stop grabbing fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            print("close device fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("destroy handle fail! ret[0x%x]" % ret)
            del self.data_buf
            sys.exit()

        del self.data_buf

    def capture_frame(self, name=None, target_dir='./', show=False):
        image = self.get_image()
        if name is not None:
            os.makedirs(target_dir, exist_ok=True)
            prefix = "capture"
            file_path = os.path.join(target_dir, prefix + name + '.jpg')
            cv2.imwrite(file_path, image)
        if show:
            cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            cv2.imshow("image", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        return image

    def init(self):
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        self.enum_device(tlayerType, deviceList)
        self.cam, self.data_buf, self.nPayloadSize = self.enable_device(0, deviceList)

if __name__ == "__main__":
    mvs_control = MVSController()
    mvs_control.init()
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 1024, 1224)
    while True:
        image = mvs_control.get_image()
        if image is not None:
            print(image.shape)
            height, width = image.shape[:2]
            center_point = (width // 2, height // 2)
            cv2.circle(image, center_point, radius=5, color=(0, 0, 255), thickness=-1)
            cv2.imshow("image", image)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    mvs_control.close_device()
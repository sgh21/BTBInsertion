import cv2
import gelsight.gsdevice as gsdevice

class GelSightController:
    def __init__(self, device_name="GelSight Mini"):
        self.device_name = device_name
        self.dev = gsdevice.Camera(self.device_name)
        self.dev.connect()
        f0 = self.dev.get_raw_image()
        print('image size = ', f0.shape[1], f0.shape[0])
        self.roi = None

    def get_raw_image(self):
        return self.dev.get_raw_image()

    def get_image(self):
        return self.dev.get_image()

    def set_roi(self, roi):
        self.roi = roi

    def get_roi_image(self):
        if self.roi:
            image = self.get_image()
            return image[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
        return self.get_image()

if __name__ == "__main__":
    gelsight = GelSightController()
    
    print('press q on image to exit, press s to save image')
    try:
        while gelsight.dev.while_condition:
            f1 = gelsight.get_image()
            # cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            f2 = cv2.resize(f1, (640, 480))
            cv2.imshow('Image', f2)
            # if cv2.waitKey(1) & 0xFF == ord('s'):
            #     cv2.imwrite(f'../result/gelsightmini/tactile_image_{i}.png', f1)
            #     print(f'Image saved_{i}')
            #     i = i + 1

            #     time.sleep(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Interrupted!')
        gelsight.dev.stop_video()

    cv2.destroyAllWindows()

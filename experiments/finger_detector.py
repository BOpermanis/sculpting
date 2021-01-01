import numpy as np
import cv2
from experiments.hand_segmentation_floodfill import HandSegmentator


class FingerDetector:
    def __init__(self, flag_3d=False):
        self.hand_segmentator = HandSegmentator()
        self.h = 480
        self.w = 640
        self.flag_3d = flag_3d
        self.window = 8
        assert self.h % self.window == 0 and self.w % self.window == 0
        self.h1 = self.h // self.window
        self.w1 = self.w // self.window
        self.depth_avg = None
        self.mask_on_desk = None

    def predict(self, rgb, depth, plane_fun=None, K=None):
        handmask = self.hand_segmentator.predict(rgb, depth, plane_fun=plane_fun, K=K)
        self.mask_on_desk = self.hand_segmentator.mask_on_desk
        self.depth_avg = self.hand_segmentator.depth_avg
        ys, xs = np.where(handmask)
        if len(ys) > 0:
            y_tresh = 40
            x_tresh = 20
            i0 = np.argmin(xs)
            y1, x1 = ys[i0], xs[i0]
            y2, x2 = y1, x1
            dx = np.abs(xs - x1)
            dy = np.abs(ys - y1)
            mask_second = np.logical_and(
                dy > y_tresh,
                dx < x_tresh
            )
            if np.any(mask_second):
                i2 = np.argmin(xs[mask_second])
                y2, x2 = ys[mask_second][i2], xs[mask_second][i2]

            return handmask, (y1, x1), (y2, x2)

        return handmask, (0, 0), (0, 0)


if __name__ == "__main__":
    from camera import RsCamera
    from utils import from_homo, plane_from_pts3d


    finger_detector = FingerDetector()

    green = np.zeros((finger_detector.h, finger_detector.w, 3), np.uint8)
    green[:, :, 1] = 255

    def overlay(rgb, mask):
        rgb[mask] = rgb[mask] // 2 + green[mask] // 2

    camera = RsCamera(flag_return_with_features=2)
    while True:
        frame = camera.get()
        rgb = frame.rgb
        depth = frame.depth

        plane_fun = None
        if frame.kp_arr is not None:
            if isinstance(frame.cloud_kp, np.ndarray):
                plane_fun = plane_from_pts3d(frame.cloud_kp)
                for kp in frame.kp_arr:
                    cv2.circle(depth, tuple(kp), 3, (0, 255, 0))

        hand_mask, (y1, x1), (y2, x2) = finger_detector.predict(rgb, depth, plane_fun=plane_fun, K=frame.K)
        kp_arr = np.array([(x1, y1), (x2, y2)]) # x un y ir otraadaak
        fingers_3d = camera.convert_depth_frame_to_pointcloud(
            finger_detector.depth_avg, kp_arr, h_target=rgb.shape[0], w_target=rgb.shape[1])[1]
        # xs, ys, zs = zip(*fingers_3d)
        # print(fingers_3d)
        finger = np.matmul(camera.cam_mat, fingers_3d.T).T
        finger = from_homo(finger)

        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        # overlay(rgb, hand_mask)
        overlay(depth, hand_mask)

        # if finger_detector.mask_on_desk is not None:
        #     mask_on_desk = cv2.resize(finger_detector.mask_on_desk.astype(np.uint8), (depth.shape[1], depth.shape[0])).astype(np.bool)
        #     overlay(rgb, mask_on_desk)

        cv2.circle(depth, (x1, y1), 1, (255, 0, 0), 10)
        cv2.circle(depth, (x2, y2), 1, (255, 0, 0), 10)

        for x, y in finger:
            if not np.isnan(x):
                cv2.circle(depth, (int(x), int(y)), 1, (0, 0, 255), 10)

        # frame = depth
        frame = np.concatenate([rgb, depth], axis=1)
        cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(10) == 27:
            break


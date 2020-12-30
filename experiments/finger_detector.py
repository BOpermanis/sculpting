import numpy as np
import cv2
from experiments.hand_segmentation_floodfill import HandSegmentator


class FingerDetector:
    def __init__(self, flag_3d=False):
        self.hand_segmentator = HandSegmentator()
        self.h = 480
        self.w = 640
        self.flag_3d = flag_3d
        self.window = 4
        assert self.h % self.window == 0 and self.w % self.window == 0
        self.h1 = self.h // self.window
        self.w1 = self.w // self.window

    def predict(self, rgb, depth):
        handmask = self.hand_segmentator.predict(rgb, depth)

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

    finger_detector = FingerDetector()

    green = np.zeros((finger_detector.h, finger_detector.w, 3), np.uint8)
    green[:, :, 1] = 255

    def overlay(rgb, mask):
        rgb[mask] = rgb[mask] // 2 + green[mask] // 2

    camera = RsCamera()
    while True:
        frame = camera.get()
        rgb = frame.rgb
        depth = frame.depth

        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50

        mask, (y1, x1), (y2, x2) = finger_detector.predict(rgb, depth)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        overlay(rgb, mask)
        overlay(depth, mask)

        cv2.circle(depth, (x1, y1), 1, (255, 0, 0), 10)
        cv2.circle(depth, (x2, y2), 1, (255, 0, 0), 10)

        # frame = np.concatenate([rgb, depth], axis=1)
        frame = depth
        cv2.imshow('video', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(10) == 27:
            break


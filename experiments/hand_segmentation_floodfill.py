import numpy as np
import cv2
from utils import numpy_avg_pathches

class HandSegmentator:
    def __init__(self):
        self.h = 480
        self.w = 640
        self.window = 4
        assert self.h % self.window == 0 and self.w % self.window == 0
        self.h1 = self.h // self.window
        self.w1 = self.w // self.window

    def predict(self, rgb, depth):
        depth = depth.astype(np.float32)
        assert rgb.shape[:2] == (self.h, self.w)
        depth[depth == 0] = np.nan

        avg = numpy_avg_pathches(depth, self.h1, self.w1)
        avg[np.isnan(avg)] = 255.0
        ih0, iw0 = np.unravel_index(avg.argmin(), avg.shape)
        m = avg.min()
        dists = np.abs(avg - m)
        d = 10
        cv2.floodFill(dists, mask=None, seedPoint=(iw0, ih0), newVal=0.0, loDiff=d, upDiff=d)
        dists = cv2.resize(dists, (self.w, self.h))
        return dists == 0


if __name__ == "__main__":
    from camera import RsCamera

    hand_segmentator = HandSegmentator()


    green = np.zeros((hand_segmentator.h, hand_segmentator.w, 3), np.uint8)
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

        mask = hand_segmentator.predict(rgb, depth)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        overlay(rgb, mask)
        overlay(depth, mask)

        # frame = np.concatenate([rgb, depth], axis=1)
        frame = depth
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break


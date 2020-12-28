import numpy as np
import cv2

def is_legit_pixel(ih, iw, h, w):
    if ih < 0 or ih > h - 1 or iw < 0 or iw > w - 1:
        return False
    return True

# clockwise
def circle_iterator(ih0, iw0, h, w, r_max):
    r = 1
    ih, iw = ih0-1, iw0-1
    while True:
        flag_atleast_on_legit_pixel = False

        while iw < iw0 + r:
            iw += 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while ih < ih0+r:
            ih += 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while iw > iw0-r:
            iw -= 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while ih > ih0-r:
            ih -= 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)
        r += 1
        ih -= 1

        if is_legit_pixel(ih, iw, h, w):
            flag_atleast_on_legit_pixel = True
            yield r, (ih, iw)

        if not flag_atleast_on_legit_pixel or r > r_max:
            break


class HandSegmentator:
    def __init__(self):
        self.h = 480
        self.w = 640
        self.window = 16
        assert self.h % self.window == 0 and self.w % self.window == 0
        self.h1 = self.h // self.window
        self.w1 = self.w // self.window

    def predict(self, rgb, depth):
        assert rgb.shape[:2] == (self.h, self.w)
        mask = np.ones((self.h1, self.w1), np.float32) * 255
        mask1 = np.zeros((self.h1, self.w1), np.uint8)
        closest_inds = None
        best_val = 255
        for ih in range(0, self.h1):
            for iw in range(0, self.w1):
                ah, bh = ih * self.window, (ih + 1) * self.window
                aw, bw = iw * self.window, (iw + 1) * self.window
                mask2 = depth[ah:bh, aw:bw] > 0
                if np.any(mask2):
                    mask[ih, iw] = np.average(depth[ah:bh, aw:bw][mask2])
                    if mask[ih, iw] < best_val:
                        closest_inds = (ih, iw)
                        best_val = mask[ih, iw]

        dist_max = 10
        flag_has_close_in_cycle = True
        r0 = 0
        mask1[closest_inds[0], closest_inds[1]] = 1
        for r, (ih, iw) in circle_iterator(closest_inds[0], closest_inds[1], self.h1, self.w1, max(self.h1, self.w1)):

            # new cycle
            if r0 != r:
                if not flag_has_close_in_cycle:
                    break
                r0 = r
                flag_has_close_in_cycle = False

            if abs(mask[ih, iw] - best_val) < dist_max:
                flag_has_close_in_cycle = True
                mask1[ih, iw] = 1

        mask1 = cv2.resize(mask1, (self.w, self.h))

        return np.stack([mask1]*3, axis=2).astype(np.bool)

if __name__ == "__main__":
    import pyrealsense2 as rs

    hand_segmentator = HandSegmentator()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    def get_frame():
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    green = np.zeros((hand_segmentator.h, hand_segmentator.w, 3), np.uint8)
    green[:, :, 1] = 255

    def overlay(rgb, mask):
        rgb[mask] = rgb[mask] // 2 + green[mask] // 2

    while True:
        rgb, depth = get_frame()
        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

        mask = hand_segmentator.predict(rgb, depth)
        overlay(rgb, mask)
        overlay(depth, mask)

        frame = np.concatenate([rgb, depth], axis=1)
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break


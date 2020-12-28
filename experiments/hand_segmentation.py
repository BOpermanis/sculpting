import numpy as np
import cv2

class HandSegmentator:
    def __init__(self):
        self.h = 480
        self.w = 640
        self.window = 32
        assert self.h % self.window == 0 and self.w % self.window == 0
        self.h1 = self.h // self.window
        self.w1 = self.w // self.window

    def predict(self, rgb, depth):
        assert rgb.shape[:2] == (self.h, self.w)
        mask = np.ones((self.h1, self.w1), np.float32) * 255
        closest_inds = None
        best_val = 255
        for ih in range(0, self.h1):
            for iw in range(0, self.w1):
                ah, bh = ih * self.window, (ih + 1) * self.window
                aw, bw = iw * self.window, (iw + 1) * self.window
                mask1 = depth[ah:bh, aw:bw] > 0
                if np.any(mask1):
                    mask[ih, iw] = np.average(depth[ah:bh, aw:bw][mask1])
                    if mask[ih, iw] < best_val:
                        closest_inds = (ih, iw)
                        best_val = mask[ih, iw]
        mask.fill(0.0)
        mask[closest_inds[0], closest_inds[1]]=1
        mask = cv2.resize(mask, (self.w, self.h))

        return np.stack([mask > 0]*3, axis=2)

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
    green[:, :, 2] = 255

    def overlay(rgb, mask):
        print(mask.shape, rgb.shape, green.shape)
        rgb[mask] = rgb[mask] // 2 + green[mask] // 2

    while True:
        rgb, depth = get_frame()
        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50
        mask = hand_segmentator.predict(rgb, depth)
        overlay(rgb, mask)
        frame = np.concatenate([
            # cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            rgb,
            cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        ], axis=1)
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break


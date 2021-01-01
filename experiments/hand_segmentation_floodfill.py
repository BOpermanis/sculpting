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
        self.depth_avg = None
        self.mask_on_desk = None
        self.u = {}
        self.v = {}
        self.x = {}
        self.y = {}
        self.K = None

    def convert_depth_frame_to_pointcloud(self, depth_image, w_target=None, h_target=None):
        height, width = depth_image.shape
        if w_target is None:
            h_target, w_target = height, width
        key = (height, width)

        if key not in self.u:
            self.u[key], self.v[key] = np.meshgrid(
                np.linspace(0, w_target - 1, width, dtype=np.int16),
                np.linspace(0, h_target - 1, height, dtype=np.int16))
            self.u[key] = self.u[key].flatten()
            self.v[key] = self.v[key].flatten()

            self.x[key] = (self.u[key] - self.K[0, 2]) / self.K[0, 0]
            self.y[key] = (self.v[key] - self.K[1, 2]) / self.K[1, 1]

        # print(depth_image.shape, width, w_target)
        z = depth_image.flatten() / 1000
        x = np.multiply(self.x[key], z)
        y = np.multiply(self.y[key], z)
        mask_legit = np.nonzero(z)

        points3d_all = np.stack([x, y, z], axis=1)

        # if kp_arr is not None:
        #     if len(kp_arr) == 0:
        #         return points3d_all[mask], []
        #     if w_target != width:
        #         kp_arr[:, 0] = width * kp_arr[:, 0] / w_target
        #         kp_arr[:, 1] = height * kp_arr[:, 1] / h_target
        #         kp_arr = kp_arr.astype(int)
        #     inds_kp = kp_arr[:, 1] * width + kp_arr[:, 0]
        #     return points3d_all[mask], points3d_all[inds_kp]

        return points3d_all, mask_legit

    def predict(self, rgb, depth, plane_fun=None, K=None):
        if K is not None and self.K is None:
            self.K = K
        depth = depth.astype(np.float32)
        assert rgb.shape[:2] == (self.h, self.w)

        depth[depth == 0] = np.nan
        avg = numpy_avg_pathches(depth, self.h1, self.w1)
        M = np.nanmax(avg)
        avg[np.isnan(avg)] = M

        self.mask_on_desk = np.zeros(avg.shape, np.bool)
        if plane_fun is not None and self.K is not None:
            pts3d, mask_legit = self.convert_depth_frame_to_pointcloud(avg, w_target=self.w, h_target=self.h)
            mask_on_desk = np.abs(pts3d[:, 1] - plane_fun(pts3d[:, (0, 2)])) < 0.05
            self.mask_on_desk = np.reshape(mask_on_desk, avg.shape)

        avg[self.mask_on_desk] = M
        if self.depth_avg is None:
            self.depth_avg = avg
        else:
            np.copyto(self.depth_avg, avg)

        ih0, iw0 = np.unravel_index(avg.argmin(), avg.shape)
        m = avg.min()
        dists = np.abs(avg - m)
        d = 10
        cv2.floodFill(dists, mask=None, seedPoint=(iw0, ih0), newVal=0.0, loDiff=d, upDiff=d)
        dists = cv2.resize(dists, (self.w, self.h))
        return dists == 0


if __name__ == "__main__":
    from camera import RsCamera
    from utils import plane_from_pts3d

    hand_segmentator = HandSegmentator()
    green = np.zeros((hand_segmentator.h, hand_segmentator.w, 3), np.uint8)
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

        hand_mask = hand_segmentator.predict(rgb, depth, K=frame.K, plane_fun=plane_fun)
        mask_on_desk = hand_segmentator.mask_on_desk.astype(np.uint8)

        mask_on_desk = cv2.resize(mask_on_desk, (depth.shape[1], depth.shape[0]), cv2.INTER_NEAREST).astype(np.bool)

        depth = (depth / 56).astype(np.uint8)
        depth[depth != 0] += 50
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)

        # overlay(rgb, hand_mask)
        overlay(depth, hand_mask)
        # overlay(depth, mask_on_desk)

        # frame = np.concatenate([rgb, depth], axis=1)
        frame = depth
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:
            break


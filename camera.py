import cv2
import numpy as np
from utils import normalize_t_shape, int2orb

class Frame:
    cnt = 0
    def __init__(self, rgb, cloud, depth, kp=None, des=None, cloud_kp=None, kp_arr=None, **kwargs):
        self.rgb = rgb
        self.cloud = cloud
        self.depth = depth
        # self.kp = kp
        self.des = des
        self.cloud_kp = cloud_kp
        self.kp_arr = kp_arr

        self.id = Frame.cnt
        Frame.cnt += 1

        # global transformation
        self.R = None
        self.t = None
        self.des2mp = None
        self.flag_global_set = False
        self.see_vector = np.asarray((0.0, 0.0, 1.0))
        for k, v in kwargs.items():
            setattr(self, k, v)

    def transform2global(self, R, t, prev_cloud_kp=None, new_inds_for_old=None, log=None):
        # assert not self.flag_global_set
        t = normalize_t_shape(t)
        self.cloud_kp = np.matmul(self.cloud_kp, R) + t
        self.flag_global_set = True
        # self.setPose(R, t)

        if log is not None:
            log['3d_point_diff'] = np.average(np.linalg.norm(self.cloud_kp[new_inds_for_old] - prev_cloud_kp, axis=1))

        if prev_cloud_kp is not None and new_inds_for_old is not None:
            self.cloud_kp[new_inds_for_old] = prev_cloud_kp


import pyrealsense2 as rs
class RsCamera:
    def __init__(self, flag_return_with_features=0):
        self.orb_params = dict(
            nfeatures=600,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            patchSize=31,
            fastThreshold=20)
        self.flag_return_with_features = flag_return_with_features
        self.width = 640
        self.height = 480

        if self.flag_return_with_features == 1:
            self.feature_extractor = cv2.ORB_create(**self.orb_params)
        if self.flag_return_with_features == 2:
            self.ncol = 6
            self.nrow = 8
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, 30)
        # config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, 30)

        cfg = self.pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        self.intr = profile.as_video_stream_profile().get_intrinsics()

        self.u = {}
        self.v = {}
        self.x = {}
        self.y = {}

        self.cam_mat = np.asarray([
            [self.intr.fx, 0, self.intr.ppx],
            [0, self.intr.fy, self.intr.ppy],
            [0, 0, 1]
        ])
        self.distCoeffs = np.zeros((8, 1), dtype=np.float32)

        # params for BA
        self.fx, self.fy, self.cx, self.cy = self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
        self.baseline = 0.06
        self.f = (self.fx + self.fy) / 2
        self.principal_point = self.cx, self.cy

    def convert_depth_frame_to_pointcloud(self, depth_image, kp_arr=None, w_target=None, h_target=None):
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

            self.x[key] = (self.u[key] - self.intr.ppx) / self.intr.fx
            self.y[key] = (self.v[key] - self.intr.ppy) / self.intr.fy

        # print(depth_image.shape, width, w_target)
        z = depth_image.flatten() / 1000
        x = np.multiply(self.x[key], z)
        y = np.multiply(self.y[key], z)
        mask = np.nonzero(z)

        points3d_all = np.stack([x, y, z], axis=1)

        if kp_arr is not None:
            if len(kp_arr) == 0:
                return points3d_all[mask], []
            if w_target != width:
                kp_arr[:, 0] = width * kp_arr[:, 0] / w_target
                kp_arr[:, 1] = height * kp_arr[:, 1] / h_target
                kp_arr = kp_arr.astype(int)
            inds_kp = kp_arr[:, 1] * width + kp_arr[:, 0]
            return points3d_all[mask], points3d_all[inds_kp]

        return points3d_all[mask, :]

    def get(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        infra_frame = frames.get_infrared_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        frame = np.asanyarray(color_frame.get_data())
        # infra_frame = np.asanyarray(infra_frame.get_data())
        # frame = cv2.cvtColor(infra_frame, cv2.COLOR_GRAY2BGR)

        ## transformation for better depth rgb correspondance for close objects
        M = np.array([
            [1.57, 0, -140],
            [0, 1.6,  -150],
        ])
        cv2.warpAffine(depth_image, M, (frame.shape[1], frame.shape[0]),
                       dst=depth_image, flags=cv2.INTER_NEAREST)
        # print(np.average(depth_image))
        des, kp_arr, kp = None, None, None
        if self.flag_return_with_features == 1:
            kp, des = self.feature_extractor.detectAndCompute(frame, mask=(depth_image > 0).astype(np.uint8) * 255)
            kp_arr = np.asarray([tuple(map(int, k.pt)) for k in kp])

        if self.flag_return_with_features == 2:
            _, corners = cv2.findChessboardCorners(frame, (self.ncol, self.nrow), None)
            is_ok = False
            if corners is not None:
                if corners.shape[0] == self.ncol * self.nrow:
                    is_ok = True
                    corners = corners.astype(int)
                    indx = np.argsort(corners[:, 0, 0])
                    indy = np.argsort(corners[:, 0, 1])
                    kp_arr = [(i, x, y) for i, (x, y) in enumerate(corners[:, 0, :]) if depth_image[y, x] > 0]
                    kp_arr = sorted(kp_arr, key=lambda x: indx[x[0]] * 1000 + indy[x[0]] * 1000)
                    kp_arr = np.asarray(kp_arr)
                    des = np.stack([int2orb(_) for _ in kp_arr[:, 0]])
                    kp_arr = kp_arr[:, 1:]

            if not is_ok:
                kp_arr = np.empty((0, 2))
                des = np.empty((0, ))

        cloud = self.convert_depth_frame_to_pointcloud(depth_image, kp_arr)

        if self.flag_return_with_features != 0 and isinstance(cloud, tuple):
            return Frame(frame, cloud[0], depth_image, kp=kp, des=des, cloud_kp=cloud[1],
                         kp_arr=kp_arr, infra=infra_frame, K=self.cam_mat)

        return Frame(frame, cloud, depth_image, K=self.cam_mat, infra=infra_frame)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    cap = RsCamera(flag_return_with_features=0)
    i_frame = 0
    while True:
        i_frame += 1
        frame = cap.get()
        # if i_frame > 100:
        #     break
        # plt.scatter(cloud[:, 2], cloud[:, 4])
        # plt.show()
        # exit()
        # print(frame.rgb_frame.shape, frame.rgb_frame.dtype, np.min(frame.rgb_frame), np.max(frame.rgb_frame))
        # print(frame.kp_arr.shape)
        # print(frame.des.shape, frame.des.dtype)
        if frame.kp_arr is not None:
            for kp in frame.kp_arr:
                cv2.circle(frame.rgb, tuple(kp), 3, (0, 255, 0))
        cv2.imshow('my webcam', frame.depth)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
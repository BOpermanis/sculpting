import numpy as np
import cv2
import cv2.aruco as aruco
from PIL import Image
from utils import from_homo, to_homo, pts_to_screen
from itertools import combinations
from camera import RsCamera
from experiments.finger_detector import FingerDetector
from utils import setup_KF

# aruco markjierim x un y ir asis, kas veidu pasu markiera plakni
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()  # Marker detection parameters

camera = RsCamera()
finger_detector = FingerDetector()

matrix_coefficients = camera.cam_mat
distortion_coefficients = camera.distCoeffs


def render_pyramid(frame, R, finger=None):
    d = 2.0
    pyramid_scale = 0.01
    pts = [
        [-1, -0.5, 3 + d],
        [-1, -0.5, 1 + d],
        [1, -0.5, 1 + d],
        [1, -0.5, 3 + d],
    ]

    pts = np.array(pts)
    # a = pts[:, 2].copy()
    # pts[:, 2] = pts[:, 1]
    # pts[:, 1] = a

    # pts = pts - np.average(pts, 0)
    pts = pts * pyramid_scale
    # pts[:, 2] += 0.04
    # exit()
    pts = to_homo(pts)
    pts = np.matmul(R, pts.T).T
    pts = from_homo(pts)
    pts_3d = pts.copy()
    # print(222, np.average(pts, 0))

    flag_point_is_in = False
    if finger is not None:
        d_max = np.max(np.linalg.norm(pts - finger))
        # print(d_max, finger, np.average(pts, 0))
        if d_max < 0.01:
            flag_point_is_in = True

    pts = pts_to_screen(matrix_coefficients, np.eye(4), pts)

    is_ok, R_to_stereo1, t, inliers = cv2.solvePnPRansac(pts_3d,
                                                         pts.astype(np.float64),
                                                         matrix_coefficients, np.zeros((4, 1)),
                                                         flags=cv2.SOLVEPNP_EPNP)

    print("these should be zeros")
    print(R_to_stereo1, t)

    h, w = frame.shape[:2]
    # pts[:, 1] = frame.shape[0] - pts[:, 1]


    color = (255, 0, 0) if flag_point_is_in else (0, 255, 0)
    for pt1, pt2 in combinations(pts, 2):
        if 0 <= pt1[0] <= w and 0 <= pt1[1] <= h and 0 <= pt2[0] <= w and 0 <= pt2[1] <= h:
            pt1, pt2 = map(lambda x: tuple(x.astype(int)), (pt1, pt2))
            cv2.line(frame, pt1, pt2, color)



frame = np.zeros((600, 600, 3), np.uint8)
render_pyramid(frame, np.eye(4))
Image.fromarray(frame).show()
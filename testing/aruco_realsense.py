import numpy as np
import cv2
import cv2.aruco as aruco
from PIL import Image
from utils import from_homo, to_homo, pts_to_screen
from itertools import combinations
from camera import RsCamera
from experiments.finger_detector import FingerDetector
from utils import setup_KF
from sklearn.linear_model import LinearRegression

# aruco markjierim x un y ir asis, kas veidu pasu markiera plakni
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()  # Marker detection parameters

camera = RsCamera()
finger_detector = FingerDetector()

matrix_coefficients = camera.cam_mat
distortion_coefficients = camera.distCoeffs


def render_pyramid(frame, R, finger=None):
    d = 5.0
    pyramid_scale = 0.01
    pts = [
        [-1, -0.5, 3 + d],
        [-1, -0.5, 1 + d],
        [1, -0.5, 1 + d],
        [1, -0.5, 3 + d],
        [0, 1, 2 + d]
    ]

    pts = np.array(pts)
    a = pts[:, 2].copy()
    pts[:, 2] = pts[:, 1]
    pts[:, 1] = a

    pts = pts - np.average(pts, 0)
    pts = pts * pyramid_scale
    pts[:, 2] += 0.04
    # exit()
    pts = to_homo(pts)
    pts = np.matmul(R, pts.T).T
    pts = from_homo(pts)
    # print(222, np.average(pts, 0))

    flag_point_is_in = False
    if finger is not None:
        d_max = np.max(np.linalg.norm(pts - finger))
        # print(d_max, finger, np.average(pts, 0))
        if d_max < 0.01:
            flag_point_is_in = True

    pts = pts_to_screen(matrix_coefficients, np.eye(4), pts)
    h, w = frame.shape[:2]
    # pts[:, 1] = frame.shape[0] - pts[:, 1]


    color = (255, 0, 0) if flag_point_is_in else (0, 255, 0)
    for pt1, pt2 in combinations(pts, 2):
        if 0 <= pt1[0] <= w and 0 <= pt1[1] <= h and 0 <= pt2[0] <= w and 0 <= pt2[1] <= h:
            pt1, pt2 = map(lambda x: tuple(x.astype(int)), (pt1, pt2))
            cv2.line(frame, pt1, pt2, color)

# frame = np.zeros((600, 600, 3), np.uint8)
# render_pyramid(frame, np.eye(4))
# Image.fromarray(frame).show()
# exit()
f1 = None
f2 = None
green = None

def enrich_with_volume(corners_3d):
    extra_pts = []
    for a, b, c in combinations(corners_3d, 3):
        d1 = b - a
        d2 = c - a
        extra_pts.append(np.cross(d1, d2) * 10 + a)
    return np.concatenate([corners_3d, np.stack(extra_pts)], 0)
    # print(corners_3d.shape)
    # print(np.stack(extra_pts))
    # print(.shape)
    # exit()


while True:
    frame = camera.get()
    frame.hand_mask = None
    if green is None:
        green = np.zeros((frame.rgb.shape[0], frame.rgb.shape[1], 3), np.uint8)
        green[:, :, 1] = 255
        def overlay(rgb, mask, green):
            rgb[mask] = rgb[mask] // 2 + green[mask] // 2

    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_BGR2GRAY)  # Change grayscale
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, arucoDict,
                                                            parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)

    if np.all(ids is not None):
        for i in range(0, len(ids)):
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            aruco.drawAxis(frame.rgb, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis

            corners_3d = camera.convert_depth_frame_to_pointcloud(frame.depth, corners[i][0, ...].astype(np.int32))[-1]

            is_ok = True
            # is_ok, R_to_stereo1, t ,inliers = cv2.solvePnPRansac(corners_3d,
            #                            corners[i][0, ...].astype(np.float64),
            #                            matrix_coefficients, np.zeros((4, 1)),
            #                            flags=cv2.SOLVEPNP_EPNP)

            # print(R_to_stereo1.flatten(), np.round(t, 2).flatten())

            if is_ok:
                R_s2c = np.eye(4)
                # R_s2c[:3, :3] = cv2.Rodrigues(R_to_stereo1)[0]
                # R_s2c[:3, 3] = t[:, 0]

                # corners_3d_2 = np.matmul(np.linalg.inv(R_s2c), to_homo(corners_3d).T).T
                # corners_3d_2 = from_homo(corners_3d_2)
                # print(np.average(corners_3d_2, 0))
                # R_c2s = np.eye(4)
                # R_c2s[:3, :3] = cv2.Rodrigues(R_to_stereo1)[0]
                # R_c2s[:3, 3] = t[:, 0]
                # R_s2c = np.linalg.inv(R_c2s)

                rvec = cv2.Rodrigues(rvec)[0]

                R_c2a = np.eye(4)
                R_c2a[:3, :3] = rvec
                R_c2a[:3, 3] = tvec

                cns = np.array([
                    [-0.01, 0.01, 0],
                    [0.01, 0.01, 0],
                    [0.01, -0.01, 0],
                    [-0.01, -0.01, 0],
                ])
                cns = to_homo(cns)
                cns = np.matmul(R_c2a, cns.T).T
                cns = from_homo(cns)
                norms_stereo = np.linalg.norm(corners_3d, axis=1)
                if len(norms_stereo > 0) == 0:
                    scale = 0.0
                else:
                    scale = np.average(np.linalg.norm(cns, axis=1)[norms_stereo > 0] / norms_stereo[norms_stereo > 0])

                print(np.average(corners_3d * scale, 0), np.average(cns, 0))

                def plane_fun(pts):
                    return np.zeros((pts.shape[0]))
                    # pts = pts - tvec[:, 0]
                    # return np.abs(np.dot(pts, z_vec[0, :]))


                hand_mask, (y1, x1), (y2, x2) = finger_detector.predict(frame.rgb, frame.depth,
                                                                        plane_fun=plane_fun,
                                                                        K=matrix_coefficients)

                fingers_3d = None
                if np.average(hand_mask) < 0.3:
                    kp_arr = np.array([(x1, y1), (x2, y2)])  # x un y ir otraadaak
                    fingers_3d = camera.convert_depth_frame_to_pointcloud(
                        finger_detector.depth_avg, kp_arr, h_target=frame.rgb.shape[0], w_target=frame.rgb.shape[1])[1]

                    if f1 is None:
                        f1 = setup_KF(fingers_3d[0, 0], fingers_3d[0, 1], fingers_3d[0, 2])
                        f2 = setup_KF(fingers_3d[1, 0], fingers_3d[1, 1], fingers_3d[1, 2])
                    else:
                        f1.predict()
                        f2.predict()
                        f1.update(fingers_3d[0, :])
                        f2.update(fingers_3d[1, :])
                elif f1 is not None:
                    f1.predict()
                    f2.predict()
                    f1.update(f1.get_pred()[:3])
                    f2.update(f2.get_pred()[:3])

                frame.finger1 = None
                frame.finger2 = None
                frame.hand_mask = None
                if fingers_3d is not None:
                    frame.finger1 = fingers_3d[0, :3].reshape(1, -1)
                    frame.finger2 = fingers_3d[1, :3].reshape(1, -1)
                    frame.hand_mask = hand_mask

                finger_3d = None
                if frame.hand_mask is not None:
                    overlay(frame.rgb, frame.hand_mask, green)
                    for finger in (frame.finger1, frame.finger2):
                        if finger is None:
                            continue

                        finger_3d = finger.copy() * scale
                        finger = np.matmul(frame.K, finger.T).T
                        finger = from_homo(finger)[0, :]

                        x, y = finger
                        if not np.isnan(x):
                            x = int(np.clip(x, 0, 639))
                            y = int(np.clip(y, 0, 479))
                            cv2.circle(frame.rgb, (x, y), 1, (255, 0, 0), 10)

                render_pyramid(frame.rgb, R_c2a, finger_3d)

    aruco.drawDetectedMarkers(frame.rgb, corners)

    cv2.imshow('frame', frame.rgb)
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):  # Quit
        break

cv2.destroyAllWindows()
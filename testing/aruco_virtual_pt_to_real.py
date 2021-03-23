import numpy as np
import cv2
import cv2.aruco as aruco
from PIL import Image
from utils import from_homo, to_homo, pts_to_screen
from itertools import combinations

# aruco markjierim x un y ir asis, kas veidu pasu markiera plakni
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()  # Marker detection parameters

s = 1.5
matrix_coefficients = np.array([
    [320 * s, 0, 320],
    [0, 240 * s, 240],
    [0, 0, 1],
], np.float32)
distortion_coefficients = np.zeros((4,1))


def render_pyramid(frame, R):
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
    # print(np.average(pts, 0).shape)
    # exit()
    pts = to_homo(pts)
    pts = np.matmul(R, pts.T).T
    pts = from_homo(pts)

    pts = pts_to_screen(matrix_coefficients, np.eye(4), pts)
    h, w = frame.shape[:2]
    # pts[:, 1] = frame.shape[0] - pts[:, 1]
    for pt1, pt2 in combinations(pts, 2):
        if 0 <= pt1[0] <= w and 0 <= pt1[1] <= h and 0 <= pt2[0] <= w and 0 <= pt2[1] <= h:
            pt1, pt2 = map(lambda x: tuple(x.astype(int)), (pt1, pt2))
            cv2.line(frame, pt1, pt2, (0, 255, 0))



# frame = np.zeros((600, 600, 3), np.uint8)
# render_pyramid(frame, np.eye(4))
# Image.fromarray(frame).show()
# exit()

cap = cv2.VideoCapture(1)  # Get the camera source

while True:
    ret, frame = cap.read()
    # print(frame.shape)
    # exit()
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
    # lists of ids and the corners beloning to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, arucoDict,
                                                            parameters=parameters,
                                                            cameraMatrix=matrix_coefficients,
                                                            distCoeff=distortion_coefficients)
    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                       distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis

            R = np.eye(4)
            R[:3, :3] = cv2.Rodrigues(rvec)[0]
            R[:3, 3] = tvec

            # print(R)
            # exit()
            render_pyramid(frame, R)

    aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):  # Quit
        break
    # exit()

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
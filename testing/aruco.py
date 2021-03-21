import numpy as np
import cv2
from PIL import Image
import os
from shutil import rmtree

import numpy as np
import cv2
import cv2.aruco as aruco

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()  # Marker detection parameters


def make_marker():
    save_dir = "/home/bruno/data"

    img_marker = cv2.aruco.drawMarker(arucoDict, 5, 100, 10)
    cv2.imwrite(save_dir + "/aruco5.png", img_marker)


def track(matrix_coefficients, distortion_coefficients):
    cap = cv2.VideoCapture(0)  # Get the camera source

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

# K = np.array([
#     [320, 0, 320],
#     [0, 240, 240],
#     [0, 0, 1],
# ], np.float32)
K = np.array([
    [640, 0, 320],
    [0, 480, 240],
    [0, 0, 1],
], np.float32)

track(K, np.zeros((4, 1)))
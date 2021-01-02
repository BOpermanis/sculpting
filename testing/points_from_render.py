import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import from_homo, to_homo, pts2d_from_render, generate_chessboard_in_camera_space, K_perfect


def check():
    with open("/home/slam_data/four_points_render.pickle", "rb") as conn:
        calibrartion_renders = pickle.load(conn)

    pts2d = []
    pts3d = []
    for render, (x, y, z) in zip(calibrartion_renders, generate_chessboard_in_camera_space()):
        pts2d.extend(pts2d_from_render(render))
        pts3d.append((x, y, z))

    pts3d = np.array(pts3d)
    pts2d = np.array(pts2d)
    print(pts3d.shape)
    print(pts2d.shape)
    (is_ok, rotation_vector, translation_vector) = cv2.solvePnP(pts3d, pts2d, K_perfect, np.zeros((4, 1)))
    R = cv2.Rodrigues(rotation_vector)[0]

    print(R)
    print(is_ok, rotation_vector, translation_vector)
    print(pts3d.shape)
    print(pts2d.shape)




if __name__ == "__main__":
    check()
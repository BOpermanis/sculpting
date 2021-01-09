import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import from_homo, to_homo, pts2d_from_render, generate_chessboard_in_camera_space, K_perfect, get_diagonal_points, get_chess2render_transformation


def check():
    # with open("/home/slam_data/four_points_render.pickle", "rb") as conn:
    #     calibrartion_renders, frame = pickle.load(conn)

    with open("/home/slam_data/four_points_render1.pickle", "rb") as conn:
        calibrartion_renders, frame = pickle.load(conn)

    with open("/home/slam_data/four_points_render2.pickle", "rb") as conn:
        calibrartion_renders, frame = pickle.load(conn)

    pts2d = []
    pts3d = []
    for render, (x, y, z) in zip(calibrartion_renders, generate_chessboard_in_camera_space()):
        pts2d.extend(pts2d_from_render(render))
        pts3d.append((x, y, z))
    pts3d = np.array(pts3d)
    pts2d = np.array(pts2d)

    T = get_chess2render_transformation(pts3d, frame)
    pts3d = np.array(list(generate_chessboard_in_camera_space()))

    A = frame.cloud_kp
    B = pts3d

    A = to_homo(A)

    A = from_homo(np.matmul(A, T))

    plt.scatter(A[:, 0], A[:, 2])
    plt.scatter(B[:, 0], B[:, 2], c="red")
    plt.show()


if __name__ == "__main__":
    check()
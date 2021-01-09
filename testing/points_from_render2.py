import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import from_homo, to_homo, pts2d_from_render, generate_chessboard_in_camera_space, K_perfect, get_diagonal_points, get_chess2render_transformation
from pprint import pprint
from scipy.spatial.transform import Rotation as random_rot

def gen_rigid_trans(seed=0):
    np.random.seed(seed)
    R = np.eye(4)
    r = random_rot.random(random_state=seed).as_matrix()
    for i in range(3):
        if r[i, i] < 0.0:
            r[i, :] *= -1.0
    R[:3, 3] = np.random.uniform(0, 5, size=(3,))
    R[:3, :3] = r
    return R

def check():

    with open("/home/slam_data/four_points_render.pickle", "rb") as conn:
        calibrartion_renders, frame = pickle.load(conn)

    pts2d = []
    pts3d = []
    for render, (x, y, z) in zip(calibrartion_renders, generate_chessboard_in_camera_space()):
        pts2d.extend(pts2d_from_render(render))
        pts3d.append((x, y, z))
    pts3d = np.array(pts3d)
    pts2d = np.array(pts2d)

    R = gen_rigid_trans()
    cloud = from_homo(np.matmul(R, to_homo(pts3d).T).T)
    cloud += np.random.normal(0, 1, size=cloud.shape) * 0.0000000000001

    f, T = get_chess2render_transformation(pts3d, cloud.copy())

    A = cloud
    B = pts3d

    A = f(A)

    plt.scatter(A[:, 0], A[:, 2])
    plt.scatter(B[:, 0], B[:, 2], c="red")
    plt.show()


def check_multiple():
    with open("/home/slam_data/four_points_render.pickle", "rb") as conn:
        calibrartion_renders, frame = pickle.load(conn)

    pts2d = []
    pts3d = []
    for render, (x, y, z) in zip(calibrartion_renders, generate_chessboard_in_camera_space()):
        pts2d.extend(pts2d_from_render(render))
        pts3d.append((x, y, z))
    pts3d = np.array(pts3d)
    pts2d = np.array(pts2d)
    B = 100

    for seed in range(B):
        R = gen_rigid_trans(seed=seed)
        cloud = from_homo(np.matmul(R, to_homo(pts3d).T).T)
        cloud += np.random.normal(0, 1, size=cloud.shape) * 0.0000000000001

        inds = np.arange(cloud.shape[0])
        log = {}
        f, T = get_chess2render_transformation(pts3d, cloud.copy(), log=log)
        inds_pred = log['indices']
        A = cloud
        B = pts3d
        A = f(A)
        if not np.all(inds == inds_pred):
            print(inds)
            print(inds_pred)

            print(np.round(T, 2))
            plt.scatter(A[:, 0], A[:, 2])
            plt.scatter(B[:, 0], B[:, 2], c="red")
            plt.show()
            exit()


if __name__ == "__main__":
    # check()
    check_multiple()
import numpy as np
from itertools import product
from utils import to_homo, from_homo
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation

h, w = 480, 640
np.random.seed(0)

def pts_to_screen(K, R, pts, flag_show=False):
    if pts.shape[1] == 3:
        pts = to_homo(pts)

    if K.shape[1] == 3:
        K = np.concatenate([K, np.zeros((3, 1))], axis=1)

    pts = np.matmul(R, pts.T).T
    pts = np.matmul(K, pts.T).T

    pts = from_homo(pts)

    if flag_show:
        img = np.zeros((h, w, 3), np.uint8)
        for x, y in pts:
            if 0 <= x <= w and 0 <= y <= h:
                x, y = map(int, (x, y))
                y = h - y
                cv2.circle(img, (x, y), 3, (0, 255, 0), 3)
        Image.fromarray(img).show()
        exit()
    return pts


def rotate_pts(R, pts):
    if pts.shape[1] == 3:
        pts = to_homo(pts)

    if R.shape[1] == 3:
        a = np.eye(4)
        a[:3, :3] = R
        R = a

    pts = np.matmul(R, pts.T).T
    pts = from_homo(pts)
    return pts


K = np.array([
    [w / 2, 0, w / 2],
    [0, h / 2, h / 2],
    [0, 0, 1]
])

# x, y, z(axis, kas iet prom no kameras)
x = product(*[np.linspace(-2, 2, num=4)] * 2)
x = np.asarray(list(x))
x = to_homo(x)
x[:, 2] = x[:, 1] + 6
x[:, 1] = -3

x_ob = np.random.normal(0, 1, size=x.shape)
x_ob += np.average(x, 0)

r_to_mesh = Rotation.from_euler('xyz', [0, 0, np.pi / 6], degrees=False).as_matrix()

x_mesh = rotate_pts(r_to_mesh, x)
x_ob_mesh = rotate_pts(r_to_mesh, x_ob)

x2d_mesh = pts_to_screen(K, np.eye(4), x_mesh)
x2d = pts_to_screen(K, np.eye(4), x)

x2d_ob_mesh = pts_to_screen(K, np.eye(4), x_ob_mesh)
x2d_ob = pts_to_screen(K, np.eye(4), x_ob)

# print(x2d.shape, x2d.dtype)
# print(x2d_mesh.shape, x2d_mesh.dtype)
# exit()
is_ok, R, t = cv2.solvePnP(x,
                           x2d_mesh.astype(np.float64),
                           K, np.zeros((4, 1)),
                           flags=cv2.SOLVEPNP_EPNP)

# is_ok, R, t = cv2.solvePnP(x_ob,
#                            x2d_ob_mesh.astype(np.float64),
#                            K, np.zeros((4, 1)),
#                            flags=cv2.SOLVEPNP_EPNP)

# print(R, R.shape)
R = cv2.Rodrigues(R)[0]
print(np.round(R, 3))
print(r_to_mesh)


# def get_trans_from
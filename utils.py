import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import warnings
from itertools import product
from sklearn.decomposition import PCA
from PIL import Image

width_perfect = 640
height_perfect = 480

K_perfect = np.array([
    [width_perfect, 0, width_perfect / 2],
    [0, height_perfect, height_perfect / 2],
    [0, 0, 1]
])

import numpy as np
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector


class rigid_transform_3D:
    def __init__(self, R=None, t=None):
        self.R = R
        self.t = t

    def get_params(self, *args, **kwargs):
        return {'R': self.R, "t": self.t}

    def set_params(self, *args, **kwargs):
        if 'R' in kwargs:
            self.R = kwargs['R']
        if 't' in kwargs:
            self.t = kwargs['t']

    def fit(self, A, B):
        assert len(A) == len(B)

        A = np.mat(A.copy().T, dtype=np.float64)
        B = np.mat(B.copy().T, dtype=np.float64)

        num_rows, num_cols = A.shape

        if num_rows != 3:
            raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # subtract mean
        if len(centroid_A.shape) == 1:
            centroid_A = np.expand_dims(centroid_A, 1)

        if len(centroid_B.shape) == 1:
            centroid_B = np.expand_dims(centroid_B, 1)

        # print("num_cols", num_cols, A.shape, centroid_A.shape, np.tile(centroid_A, (1, num_cols)).shape)
        # Am = A - tile(centroid_A, (1, num_cols))
        # Bm = B - tile(centroid_B, (1, num_cols))

        Am = A - centroid_A
        Bm = B - centroid_B

        # H = matmul(Am, transpose(Bm))
        H = np.matmul(Am, Bm.T)
        # H = Am * transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T * U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            # print("det(R) < 0, reflection detected!, correcting for it ...\n");
            Vt[2, :] *= -1
            R = Vt.T * U.T

        # t = -R * centroid_A + centroid_B

        t = -np.matmul(R, centroid_A) + centroid_B

        self.R = R
        if len(t.shape) == 1:
            self.t = np.expand_dims(t, 1)
        else:
            self.t = t

    def score(self, A, B):
        pred = self.predict(A)
        return sum(np.linalg.norm(B - pred, axis=1))

    def predict(self, A):
        n = A.shape[0]
        A = A.copy().T
        # exit()
        return (np.matmul(self.R, A) + np.tile(self.t, (1, n))).T
        # return matmul(self.R, A).T + tile(self.t, (1, n))

def to_homo(arr):
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)

def from_homo(arr):
    return arr[:, :-1] / arr[:, -1:]

def initialize_transformation_chessboard2mesh(frame, flag_return_intermediate_results=False):

    scale = 0.3
    scale_half = scale / 2.0
    x_center, y_center, z_center = 0, -0.5, -1.5

    dists = distance_matrix(frame.cloud_kp, frame.cloud_kp)
    inds = np.argsort(-dists.flatten())

    # stuuru indeksi
    ind_corners, _ = np.unravel_index(inds[:4], dists.shape)
    a1, a2, b1, b2 = ind_corners
    r = (dists[a1, b1] + dists[a2, b2]) / (dists[a1, b2] + dists[a2, b1])
    if r < 1.0:
        r = 1 / r

    target_corners = np.asarray([
        (x_center - scale_half, y_center, z_center - scale_half * r), # a1
        (x_center - scale_half, y_center, z_center + scale_half * r), # a2
        (x_center + scale_half, y_center, z_center + scale_half * r), # b1
        (x_center + scale_half, y_center, z_center - scale_half * r), # b2
    ])

    corners = frame.cloud_kp[ind_corners, :]

    # ax = b, taatad no chessboard to mesh
    try:
        P = np.linalg.solve(to_homo(corners), to_homo(target_corners))
    except np.linalg.LinAlgError:
        P = None

    if flag_return_intermediate_results:
        return P, ind_corners, r
    return P

def project_to_camera(pts3d, R, K):
    a = np.matmul(R, to_homo(pts3d).T)
    a = np.matmul(K, a)
    a = from_homo(a.T)
    return a

def setup_KF(x, y, z=None):
    flag_2d = z is None
    if flag_2d:
        kf = KalmanFilter(dim_x=4, dim_z=2)
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]

        ])

        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        kf.x[0] = x
        kf.x[1] = y
    else:
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ])
        kf.x[0] = x
        kf.x[1] = y
        kf.x[2] = z
        # kf.R[2:, 2:] *= 10.
        # kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        # kf.P *= 10.
        # kf.Q[-1, -1] *= 0.01
        # kf.Q[4:, 4:] *= 0.01
    kf.B = np.eye(kf.dim_u)
    kf.get_pred = lambda: np.dot(kf.F, kf.x)
    return kf


def is_legit_pixel(ih, iw, h, w):
    if ih < 0 or ih > h - 1 or iw < 0 or iw > w - 1:
        return False
    return True

# clockwise
def circle_iterator(ih0, iw0, h, w, r_max):
    r = 1
    ih, iw = ih0-1, iw0-1
    while True:
        flag_atleast_on_legit_pixel = False

        while iw < iw0 + r:
            iw += 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while ih < ih0+r:
            ih += 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while iw > iw0-r:
            iw -= 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)

        while ih > ih0-r:
            ih -= 1
            if is_legit_pixel(ih, iw, h, w):
                flag_atleast_on_legit_pixel = True
                yield r, (ih, iw)
        r += 1
        ih -= 1

        if is_legit_pixel(ih, iw, h, w):
            flag_atleast_on_legit_pixel = True
            yield r, (ih, iw)

        if not flag_atleast_on_legit_pixel or r > r_max:
            break


def numpy_avg_pathches_search(img, h1, w1, reshape_fun, avg_fun):
    h, w = img.shape[:2]
    assert h % h1 == 0 and w % w1 == 0 and h // h1 == w // w1
    window = h // h1
    img = reshape_fun(img)
    img = avg_fun(img)
    return img


def numpy_avg_pathches(img, h1, w1):
    h, w = img.shape[:2]
    flag_rgb = len(img.shape) == 3
    assert h % h1 == 0 and w % w1 == 0 and h // h1 == w // w1
    window = h // h1

    if flag_rgb:
        img = np.reshape(img, (h1, window, w1, window, 3))
    else:
        img = np.reshape(img, (h1, window, w1, window))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = np.nanmean(img, (1, 3))
    return img


def normalize_t_shape(t):
    if len(t.shape) > 1:
        return t[:, 0]
    else:
        return t


def int2orb(i):
    np.random.seed(i)
    return np.random.randint(256, size=32).astype(np.uint8)

def plane_from_pts3d(pts3d):
    lr = LinearRegression()
    lr.fit(pts3d[:, (0, 2)], pts3d[:, 1])
    return lr.predict

def generate_chessboard_in_camera_space():
    r = 0.75
    s = 0.5
    s2 = s / 2
    x, y, z = 0.0, -0.5, -1.5
    return product(np.linspace(x - s2, x + s2, num=8), [y], np.linspace(z - s2, z + s2, num=6))
    # zs, ys, xs = zip(*list(product(np.linspace(z - s2, z + s2, num=6), [y], np.linspace(x - s2, x + s2, num=8))))
    # return zip(xs, ys, zs)


# l = list(generate_chessboard_in_camera_space())
# print(len(l), len(set(l)))
# exit()

def pts2d_from_render(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, img = cv2.threshold(img, 127, 255, 0)
    *_, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    pts = []
    for c in contours:
        pts.append(np.average(c[:, 0, :], axis=0))
    return pts



def get_diagonal_points(pts):
    dists = distance_matrix(pts, pts)
    indsa, indsb = np.unravel_index(np.argsort(-dists.flatten()), dists.shape)

    pairs = set()
    unique_corners = set()
    for a, b in zip(indsa, indsb):
        if a not in unique_corners and b not in unique_corners:
            if a > b:
                pairs.add((a, b))
            else:
                pairs.add((b, a))
            unique_corners.add(a)
            unique_corners.add(b)

        if len(pairs) == 2:
            break

    pairs = list(pairs)
    a1, a2 = pairs[0]
    b1, b2 = pairs[1]
    # print(np.unravel_index(np.argsort(-dists.flatten())[:4], dists.shape))
    # print(np.sort(-dists.flatten())[:5])
    # print(a1, a2, b1, b2)
    # import matplotlib.pyplot as plt
    # print(len(set([tuple(a) for a in pts])))
    # plt.scatter(pts[:, 0], pts[:, 2])
    # plt.scatter(pts[a1, 0], pts[a1, 2], c="red")
    # plt.scatter(pts[a2, 0], pts[a2, 2], c="red")
    # plt.scatter(pts[b1, 0], pts[b1, 2], c="green")
    # plt.scatter(pts[b2, 0], pts[b2, 2], c="green")
    # plt.show()
    # exit()

    g1 = np.linalg.norm(pts[a1] - pts[b1]) + np.linalg.norm(pts[b2] - pts[a2])
    g2 = np.linalg.norm(pts[a1] - pts[a2]) + np.linalg.norm(pts[b1] - pts[b2])
    # (.., ..), (.., ..) - paari veido garaakaas malas (abaam malaam ir viens un tas pats virziens)
    if g1 > g2:
        a1, a2, b1, b2 = a1, b1, b2, a2

    # taisnstuura virzieni
    vec_major = ((pts[a1, :] - pts[a2, :]) + (pts[b1, :] - pts[b2, :])) / 2
    vec_minor = ((pts[a1, :] - pts[b1, :]) + (pts[a2, :] - pts[b2, :])) / 2
    # print(np.dot(vec_minor, vec_major))
    # print(vec_minor, vec_major)
    # exit()

    return vec_major, vec_minor, [a1, a2, b1, b2]


def get_lr_coefs_non_singular(X, Y):
    # apreekjinu preciizaaku transformaaciju izmantojot visus taisnstuura punktus
    lr = LinearRegression(fit_intercept=False)
    X = to_homo(X)
    Y = to_homo(Y)
    lr.fit(X, Y)
    T = np.eye(4)
    T[:, :3] = lr.coef_.T
    return lr.coef_.T, lr.score(X, Y)
    # return T, np.sum(np.abs(np.eye(4) - T))

class get_lr_coefs:
    def __init__(self, a, b):
        a = np.array(a)[:, :3]
        b = np.array(b)[:, :3]

        self.pca_a = PCA(n_components=3)
        self.pca_b = PCA(n_components=3)

        self.pca_a.fit(a)
        # self.pca_a.components_[0, :] *= -1

        self.pca_b.fit(b)
        # self.pca_b.components_[:, 2] *= -1

        # print(np.round(self.pca_a.components_, 2))
        # exit()
        a1 = self.pca_a.transform(a)
        self.mask_a_ok = self.pca_a.explained_variance_ratio_ > 0.000001

        b1 = self.pca_b.transform(b)

        mask_b_ok = self.pca_b.explained_variance_ratio_ > 0.000001

        a2 = a1[:, self.mask_a_ok]
        b2 = b1[:, mask_b_ok]

        self.lr = LinearRegression()
        self.lr.fit(a2, b2)
        # print(np.round(self.lr.coef_, 2))
        # print(np.round(self.pca_a.components_, 2))
        # print(np.round(self.pca_b.components_, 2))
        # exit()

    def transform(self, a):
        pred = self.lr.predict(self.pca_a.transform(a)[:, self.mask_a_ok])
        if pred.shape[1] == 2:
            pred = np.concatenate([pred, np.zeros((pred.shape[0], 1))], 1)
        return self.pca_b.inverse_transform(pred)

def get_chess2render_transformation(cloud, pts3d, log=None):
    # balstoties uz taisnstuura gjeometriskajaam iipashiibaam
    vec_render_major, vec_render_minor, inds_render = get_diagonal_points(pts3d)
    vec_chess_major, vec_chess_minor, inds_chess = get_diagonal_points(cloud)
    # v_render = np.cross(vec_render_major, vec_render_minor)
    # v_chess = np.cross(vec_chess_major, vec_chess_minor)

    dist_mat = distance_matrix(cloud[inds_chess, :].copy(), pts3d[inds_render, :].copy())
    row_ind, col_ind = linear_sum_assignment(dist_mat)

    inds_chess = np.array(inds_chess)[row_ind]
    inds_render = np.array(inds_render)[col_ind]

    T1 = get_lr_coefs(cloud[inds_chess, :].copy(), pts3d[inds_render, :].copy())
    cloud1 = T1.transform(cloud)

    # nodibinu 1:1 attieciibas
    dist_mat = distance_matrix(cloud1, pts3d)
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    indices = col_ind[row_ind]

    return indices


def pts_to_screen(K, R, pts, flag_show=False):
    if pts.shape[1] == 3:
        pts = to_homo(pts)

    if K.shape[1] == 3:
        K = np.concatenate([K, np.zeros((3, 1))], axis=1)

    pts = np.matmul(R, pts.T).T
    pts = np.matmul(K, pts.T).T

    pts = from_homo(pts)

    if flag_show:
        h, w = 480, 640
        img = np.zeros((h, w, 3), np.uint8)
        for x, y in pts:
            if 0 <= x <= w and 0 <= y <= h:
                x, y = map(int, (x, y))
                y = h - y
                cv2.circle(img, (x, y), 3, (0, 255, 0), 3)
        Image.fromarray(img).show()
        exit()
    return pts


if __name__ == "__main__":
    from itertools import permutations

    def gt_avg(img, h1, w1, window):
        avg = np.zeros((h1, w1))
        for ih in range(0, h1):
            for iw in range(0, w1):
                ah, bh = ih * window, (ih + 1) * window
                aw, bw = iw * window, (iw + 1) * window
                avg[ih, iw] = np.average(img[ah:bh, aw:bw])
        return avg

    window = 10
    h, w = 50, 60
    h1, w1 = h // window, w // window
    img = np.random.uniform(0, 1, (h, w))
    gt = gt_avg(img, h1, w1, window)
    for a, b, c, d in permutations((h1, w1, window, window), ):
        for d1, d2 in permutations((0, 1, 2, 3), 2):

            def reshape_fun(arr):
                return np.reshape(arr, (a, b, c, d))

            def avg_fun(arr):
                return np.nanmean(arr, (d1, d2))


            # pred = numpy_avg_pathches_search(img, h1, w1, reshape_fun, avg_fun)
            try:
                pred = numpy_avg_pathches_search(img, h1, w1, reshape_fun, avg_fun)
            except Exception as e:
                print(e)
                pred = 0

            if gt.shape == pred.shape:
                if np.sum(np.abs(gt - pred)) < 0.00001:
                    print(np.sum(np.abs(gt - pred)))
                    print(a, b, c, d, d1, d2)
                    exit()




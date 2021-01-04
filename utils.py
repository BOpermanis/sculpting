import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment
import warnings
from itertools import product

width_perfect = 640
height_perfect = 480

K_perfect = np.array([
    [width_perfect, 0, width_perfect / 2],
    [0, height_perfect, height_perfect / 2],
    [0, 0, 1]
])

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
    assert h % h1 == 0 and w % w1 == 0 and h // h1 == w // w1
    window = h // h1
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
    indsa, indsb = np.unravel_index(np.argsort(-dists.flatten())[:4], dists.shape)

    pairs = set()
    for a, b in zip(indsa, indsb):
        if a > b:
            pairs.add((a, b))
        else:
            pairs.add((b, a))

    pairs = list(pairs)
    a1, a2 = pairs[0]
    b1, b2 = pairs[1]
    # print(pairs)
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
    g2 = np.linalg.norm(pts[a1] - pts[b2]) + np.linalg.norm(pts[b1] - pts[a2])

    # (.., ..), (.., ..) - paari veido garaakaas malas (abaam malaam ir viens un tas pats virziens)
    if g1 > g2:
        a1, a2, b1, b2 = a1, b1, b2, a2
    else:
        a1, a2, b1, b2 = a1, b2, b1, a2

    # taisnstuura virzieni
    vec_major = ((pts[a1, :] - pts[a2, :]) + (pts[b1, :] - pts[b2, :])) / 2
    vec_minor = ((pts[a1, :] - pts[b1, :]) + (pts[a2, :] - pts[b2, :])) / 2

    return vec_major, vec_minor


def get_chess2render_transformation(calibrartion_renders, frame):
    pts2d = []
    pts3d = []
    for render, (x, y, z) in zip(calibrartion_renders, generate_chessboard_in_camera_space()):
        pts2d.extend(pts2d_from_render(render))
        pts3d.append((x, y, z))
    pts3d = np.array(pts3d)
    pts2d = np.array(pts2d)

    x = frame.cloud_kp.copy()

    # balstoties uz taisnstuura gjeometrikajaam iipashiibaam
    vec_render_major, vec_render_minor = get_diagonal_points(pts3d)
    vec_chess_major, vec_chess_minor = get_diagonal_points(x)

    v_render = np.cross(vec_render_major, vec_render_minor)
    v_chess = np.cross(vec_chess_major, vec_chess_minor)

    m_render = np.stack([vec_render_major, vec_render_minor, v_render])
    m_chess = np.stack([vec_chess_major, vec_chess_minor, v_chess])

    T1 = np.linalg.solve(m_chess, m_render)
    x = np.matmul(frame.cloud_kp, T1)

    # nodibinu 1:1 attieciibas
    dist_mat = distance_matrix(x, pts3d)
    row_ind, col_ind = linear_sum_assignment(dist_mat)
    indices = col_ind

    # apreekjinu preciizaaku transformaaciju izmantojot visus taisnstuura punktus
    lr = LinearRegression(fit_intercept=False)
    lr.fit(to_homo(x), to_homo(pts3d[indices, :]))
    # lr.fit(x, pts3d[indices, :])

    # x = np.matmul(x, lr.coef_)
    # plt.scatter(x[:, 0], x[:, 2])
    # plt.scatter(pts3d[:, 0], pts3d[:, 2], c="red")
    # plt.show()
    # exit()

    # T2 = np.eye(4)
    # T2[:3, :3] = lr.coef_.T

    T = np.eye(4)
    T[:3, :3] = T1
    # T[:3, 3] = t
    T = np.matmul(T, lr.coef_.T)
    return T


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




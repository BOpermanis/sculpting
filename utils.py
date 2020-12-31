import numpy as np
from filterpy.kalman import KalmanFilter

def to_homo(arr):
    return np.concatenate([arr, np.ones((arr.shape[0], 1))], axis=1)

def from_homo(arr):
    return arr[:, :-1] / arr[:, -1:]

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




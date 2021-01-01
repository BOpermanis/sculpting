import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import from_homo, to_homo, initialize_transformation_chessboard2mesh
from sklearn.decomposition import PCA
from scipy.spatial import distance_matrix


def check():
    with open("/home/slam_data/temp.pickle", "rb") as conn:
        frame = pickle.load(conn)

    P, ind_corners, r = initialize_transformation_chessboard2mesh(frame, flag_return_intermediate_results=True)
    print(np.linalg.det(P))
    print(np.round(P, 2))
    pca = PCA(n_components=2)
    pca.fit(frame.cloud_kp)

    y = pca.transform(frame.cloud_kp)
    plt.scatter(y[:, 0], y[:, 1])
    plt.scatter(y[ind_corners, 0], y[ind_corners, 1], c="red")
    plt.show()



if __name__ == "__main__":
    check()
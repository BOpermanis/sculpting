import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from utils import get_lr_coefs

a = [[ 0.25, -0.5,  -1.25,  1.  ],
 [-0.25, -0.5,  -1.75,  1.  ],
 [ 0.25, -0.5,  -1.75,  1.  ],
 [-0.25, -0.5,  -1.25,  1.  ]]
b = [[ 0.25, -0.5,  -1.25,  1.  ],
 [-0.25, -0.5,  -1.75,  1.  ],
 [ 0.25, -0.5,  -1.75,  1.  ],
 [-0.25, -0.5,  -1.25,  1.  ]]

a = np.array(a)[:, :3]
b = np.array(b)[:, :3]

ob = get_lr_coefs(a, b)

print(ob.transform(a))
print(b)

# pca_a = PCA(n_components=3)
# pca_b = PCA(n_components=3)
#
# a1 = pca_a.fit_transform(a)
# mask_a_ok = pca_a.explained_variance_ratio_ > 0.000001
#
#
# b1 = pca_b.fit_transform(b)
# mask_b_ok = pca_b.explained_variance_ratio_ > 0.000001
#
# a2 = a1[:, mask_a_ok]
# b2 = b1[:, mask_b_ok]
#
# lr = LinearRegression()
# lr.fit(a2, b2)
#
# pred = lr.predict(pca_a.transform(a)[:, mask_a_ok])
# pred = np.concatenate([pred, np.zeros((pred.shape[0], 1))], 1)
# pred = pca_b.inverse_transform(pred)
# print(pred)
# print(b)

# print(lr.coef_)

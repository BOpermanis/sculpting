import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

np.random.seed(0)
N = 20
x = np.random.normal(0, 1, (N, 2))
# x1 = x + np.random.normal(0, 1, (N, 2))
inds = np.random.choice(N, N, replace=False)
x1 = x[inds]

dists = distance_matrix(x1, x)
row_ind, col_ind = linear_sum_assignment(dists)

print(inds)
print(row_ind)
print(col_ind)
print(col_ind[row_ind])
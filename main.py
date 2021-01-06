import scipy.io
import numpy as np

loaded = scipy.io.loadmat('FaceData.mat')
mat: np.ndarray = loaded['FaceData']
shape=mat.shape
mat=mat.reshape((shape[0]*shape[1]))
print(mat.shape)
mat=np.asarray([m[0] for m in mat]) # remove parenthesis
print(mat[0].shape)

def compute_mean(mat):
    m = 0
    for col in mat:
        m += np.mean(col)
    return m / mat.shape[0]

theta_0=(compute_mean(mat))



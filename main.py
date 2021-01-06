import scipy.io
import numpy as np
import matplotlib.pyplot as plt
loaded = scipy.io.loadmat('FaceData.mat')
mat: np.ndarray = loaded['FaceData']
shape = mat.shape
mat = mat.flatten()
print(mat.shape)
image_shape=mat[0][0].shape
plt.imshow(mat[0][0])
plt.show()
print(image_shape)
mat = np.asarray([m[0].flatten() for m in mat])  # remove parenthesis
print(mat[0].shape)


def compute_mean(mat):
    m = 0
    for col in mat:
        m += np.mean(col)
    return m / mat.shape[0]  # np.mean(mat)


theta_0 = compute_mean(mat)

x_0 = np.asarray([m - theta_0 for m in mat])
n = x_0.shape[0]


def get_covariance_matrix(x_0, n):
    m = []
    for col in x_0:
        m.append(np.dot(col, col.transpose()))
    return np.asarray(m) * (1 / (n - 1))


get_covariance_matrix(x_0, n)
cov = 1 / (n - 1) * np.dot(x_0.transpose(), x_0)
print(cov.shape)
eig_value, eig_vector = np.linalg.eig(cov)

for i in range(10):
    img=eig_vector[i].real+mat[i]
    plt.imshow(img.reshape(image_shape))
    plt.show()
    plt.imshow(mat[i].reshape(image_shape))
    plt.show()
    plt.imshow(eig_vector[i].real.reshape(image_shape))
    plt.show()



import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

loaded = scipy.io.loadmat('FaceData.mat')
matx: np.ndarray = loaded['FaceData']

# create a matrix 'mat' with dimensions dxn with d = pxq and n = number of pictures
shape = matx.shape
matx = matx.flatten()
print(matx.shape)
image_shape = matx[0][0].shape
d = image_shape[0] * image_shape[1]
# plt.imshow(matx[0][0])
# 
print(image_shape)
matx = np.asarray([m[0].flatten() for m in matx])  # remove parenthesis
mat = np.transpose(matx)
print(mat.shape)


# calculate the mean
def compute_mean(mat):
    m = 0
    for col in mat:
        m += np.mean(col)
    return m / mat.shape[0]  # np.mean(mat)


theta_0 = compute_mean(mat)
print("theta", theta_0)

# Create a training matrix
x_0 = np.asarray([m - theta_0 for m in mat])
print("xo shape", x_0.shape)

# n is number of figures
n = x_0.shape[1]
print(n)

# Create covariance matrix with dimensions dxd (2576 x 2576)
cov = (1 / (n - 1)) * np.dot(x_0, x_0.transpose())
# print(cov)
print(cov.shape)

# Compute eigenvectors and eigenfaces
eig = la.eig(cov)
# Print eigenvalues
eig_value = eig[0]
# print(eig_value)
# print eigenvectors
eig_vector = eig[1]
print(eig_vector)
print(eig_vector.shape)

# reshape vectors
eigen_face = eig_vector
print(eigen_face.shape)

# show an eigenface
plt.figure(1)
plt.imshow(eigen_face[5,].real.reshape(image_shape))


# show an eigenface over image
plt.figure(2)
plt.imshow(matx[0].reshape(image_shape))


plt.figure(3)
plt.imshow(matx[0].reshape(image_shape))
plt.imshow(eigen_face[0,].real.reshape(image_shape))


for i in range(2):
    plt.figure(4 + i * 3)
    img = eigen_face[i,].real * 255 + matx[i]
    plt.imshow(img.reshape(image_shape))
    plt.title("Addition of both")

    plt.figure(4 + i * 3 + 1)
    img = matx[i] - eigen_face[i,].real * 255
    plt.imshow(img.reshape(image_shape))
    plt.title("Difference of both")

    
    plt.figure(4 + i * 3 + 2)
    plt.imshow(matx[i].reshape(image_shape))
    plt.title("face")
    
    plt.figure(4 + i * 3 + 3)
    plt.imshow(eigen_face[i].real.reshape(image_shape))
    plt.title("eigen face")
    

# Question 7
import random

m = random.randint(1, d)  # [1,d)


def extract_m_largest(m):
    # listing all the eigen value, with their respecting indexes
    eig_temp = [(v.real, i) for i, v in enumerate(eig_value)]
    # sorting the eigen values
    eig_temp.sort(key=lambda x: x[0], reverse=True)
    # m first largest eigen values
    eig_temp = eig_temp[:m]
    # extract phiM eigenvectors with the m largest eigenvalues
    eig_vector_extracted = [eig_vector[i] for _, i in eig_temp]
    print(len(eig_vector_extracted), m)
    return eig_vector_extracted


# Question 3.3.1
from sklearn.model_selection import train_test_split

data, labels = matx, range(len(matx))

data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.5, random_state=42) \

def v(_m):
    return sum(eig_value[:_m].real) / sum(eig_value.real)


v_m = [v(_m) for _m in range(d)]

plt.figure(15)
plt.plot(range(d), v_m)


plt.figure(16)
plt.plot(range(500), v_m[:500])


plt.show()
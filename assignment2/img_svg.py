# svd 奇异值分解
import numpy as np
from PIL import Image


def sort_eigenvector(value, vector):
    idx = value.argsort()[::-1]
    value = value[idx]
    vector = vector[:, idx]
    return vector


def solve_svg(A):

    flag = False
    if A.shape[0] > A.shape[1]:
        A = A.T
        flag = True
    m, n = A.shape[0], A.shape[1]
    # 1. 求ATA的特征值和特征向量
    MA = A.T @ A
    # print(MA)
    # 2. 特征向量单位化，得到n阶正交矩阵V
    eigenvalues0, eigenvector0 = np.linalg.eigh(MA)
    print(len(eigenvalues0))
    eigenvector0 = sort_eigenvector(eigenvalues0, eigenvector0)
    # 3. 计算奇异值，得到m×n矩形对角矩阵Σ
    eigenvalues0 = np.sort(eigenvalues0[eigenvalues0 > 0])[::-1]
    # print(eigenvalues0)
    # print(eigenvector0)

    Sigma = np.sqrt(np.diag(eigenvalues0))

    sm, sn = Sigma.shape[0], Sigma.shape[1]
    if m > sm:
        Sigma = np.r_[Sigma, np.zeros((m-sm, sn))]
    else:
        Sigma = Sigma[:m, :]
    if n > sn:
        Sigma = np.c_[Sigma, np.zeros((m, n-sn))]

    # 4. 求m阶正交矩阵U
    # 这么求是错误的，存在U和V的特征向量正负不匹配问题，应当从上面入手求
    # AM = A@A.T
    # eigenvalues1, eigenvector1 = np.linalg.eigh(AM)
    # eigenvector1 = sort_eigenvector(eigenvalues1, eigenvector1)

    # U=AVS^-1
    U = np.mat(np.zeros((m, m)))
    for j in range(m):
        U[:, j] = (1/np.sqrt(eigenvalues0[j]) * A @
                   eigenvector0[:, j]).reshape(-1, 1)
    if flag:
        return eigenvector0, Sigma, U.T
    else:
        return U, Sigma, eigenvector0.T
    # return U, Sigma, eigenvector0


# def test(A):
#     m, n = A.shape[0], A.shape[1]
#     import numpy.linalg as la
#     U, S, V = la.svd(A)
#     print('√'*100)
#     print(U)
#     S = np.diag(S)
#     sm, sn = S.shape[0], S.shape[1]

#     S = np.r_[S, np.zeros((m-sm, sn))]
#     S = np.c_[S, np.zeros((m, n-sn))]
#     print(S)
#     print(V)
#     print(U*S*V)


def image_svd(image, n):
    U, S, V = solve_svg(image)
    r = U[:, :n].dot(S[:n, :n]).dot(V[:n, :]) * 255.0

    return r


def image_compress(image, n):
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    shape = image.shape

    result = np.zeros(shape)
    result[:, :, 0] = image_svd(r, n)
    result[:, :, 1] = image_svd(g, n)
    result[:, :, 2] = image_svd(b, n)
    return result


def generate_new_image(img_arr, k):
    p = image_compress(img_arr, k)
    p = Image.fromarray(p.astype('uint8'))
    p.save(f"k_{k}.png")
    print(f"finish k={k}")


if __name__ == '__main__':
    img_arr = np.array(Image.open('flower.jpg')) / 255.0
    generate_new_image(img_arr, 20)
    generate_new_image(img_arr, 35)

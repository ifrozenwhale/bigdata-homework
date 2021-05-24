# svd 奇异值分解
import numpy as np


def sort_eigenvector(value, vector):
    idx = value.argsort()[::-1]
    value = value[idx]
    vector = vector[:, idx]
    return vector


def solve(A):
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

    eigenvector0 = sort_eigenvector(eigenvalues0, eigenvector0)
    # 3. 计算奇异值，得到m×n矩形对角矩阵Σ
    eigenvalues0 = np.sort(eigenvalues0[eigenvalues0 > 0])[::-1]
    # print(eigenvalues0)
    print(eigenvector0)

    Sigma = np.sqrt(np.diag(eigenvalues0))

    sm, sn = Sigma.shape[0], Sigma.shape[1]

    Sigma = np.r_[Sigma, np.zeros((m-sm, sn))]
    Sigma = np.c_[Sigma, np.zeros((m, n-sn))]

    # 4. 求m阶正交矩阵U
    # 这么求是错误的，存在U和V的特征向量正负不匹配问题，应当从上面入手求
    # AM = A@A.T
    # eigenvalues1, eigenvector1 = np.linalg.eigh(AM)
    # eigenvector1 = sort_eigenvector(eigenvalues1, eigenvector1)

    # U=AVS^-1
    U = np.mat(np.zeros((m, m)))
    for j in range(len(eigenvalues0)):
        U[:, j] = (1/np.sqrt(eigenvalues0[j]) * A @
                   eigenvector0[:, j]).reshape(-1, 1)
    if flag:
        return U.T, Sigma, eigenvector0
    else:
        return U, Sigma, eigenvector0.T
    # return U, Sigma, eigenvector0


def test(A):
    m, n = A.shape[0], A.shape[1]
    import numpy.linalg as la
    U, S, V = la.svd(A)
    print('√'*100)
    print(U)
    S = np.diag(S)
    sm, sn = S.shape[0], S.shape[1]

    S = np.r_[S, np.zeros((m-sm, sn))]
    S = np.c_[S, np.zeros((m, n-sn))]
    print(S)
    print(V)
    print(U*S*V)


if __name__ == '__main__':
    A = np.mat([[1, 2, 0], [2, 0, 2]])
    # A = np.array([[1, 1], [2, 2], [0, 0]])
    U, Sigma, V = solve(A)
    print(U)
    print(Sigma)
    print('V')
    print(V.T)
    # print(U@Sigma@V.T)
    # test(A.T)

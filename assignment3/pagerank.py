import numpy as np
import networkx as nx


def pagerank(A, alpha, beta):
    # print(np.sum(A, axis=1))
    D = np.diagflat(np.sum(A, axis=1))
    I = np.identity(4)
    one_vec = np.ones((4, 1))

    # Cp = beta * np.linalg.inv(I - alpha *
    #                           (np.dot(A.T, np.linalg.inv(D)))) * one_vec
    Cp = np.ones((4, 1))
    for i in range(100):
        Cp = alpha * np.dot(np.dot(A.T, np.linalg.inv(D)), Cp) + beta * one_vec
    print(Cp)


if __name__ == '__main__':
    A = np.matrix([[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]])
    pagerank(A, 1, 0)
    pagerank(A, 0.85, 1)
    pagerank(A, 0, 1)

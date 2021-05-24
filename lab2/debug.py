import pandas as pd
import numpy as np


def init_P_Q_set(K=5):

    df = pd.read_table("data/ratings.dat", encoding='utf-8',
                       header=None, sep='::', engine='python')
    df.columns = ['UserID', 'MovieID', 'Rating', 'TimeStamp']

    # 打乱顺序
    df = df.sample(frac=1)

    n_User = df['UserID'].max()
    n_Movie = df['MovieID'].max()

    P = np.random.randint(1, 3, (n_User+1, K)) / (K**0.5)
    Q = np.random.randint(1, 3, (K, n_Movie+1)) / (K**0.5)

    # P = np.random.randn(n_User+1, K)
    # Q = np.random.randn(K, n_Movie)

    grouped = df.groupby('UserID')

    train_set = {}
    val_set = {}
    test_set = {}

    for u, group in grouped:
        train_set[u] = {}
        val_set[u] = {}
        test_set[u] = {}

        i = 0
        for v, w in zip(group['MovieID'], group['Rating']):
            if i < 0.8*len(group):
                train_set[u][v] = w
            elif i < 0.9*len(group):
                val_set[u][v] = w
            else:
                test_set[u][v] = w
            i += 1
    return P, Q, train_set, test_set, val_set

# def optimize(P, Q, learning_rate, train_set):
#     Pn = np.array(P, copy=True)
#     Qn = np.array(Q, copy=True)
#
#     U, I, K = len(P), len(Q[0]), len(P[0])
#
#     cost = 0
#     n = 0
#
#     # e_ui_map = {}
#
#     for u in range(1, U):
#
#         # e_ui_map[u] = {}
#
#         for i in range(1, I):
#             if u in train_set.keys() and i in train_set[u].keys():
#                 e_ui = train_set[u][i] - np.dot(P[u, :], Q[:, i])
#
#                 # e_ui_map[u][i] = e_ui
#
#                 # print(P[u, :])
#                 # print(Q[:, i])
#                 # print(e_ui)
#
#                 cost += e_ui**2
#                 n += 1
#
#                 for k in range(0, K):
#                     Pn[u][k] = P[u][k] + learning_rate * e_ui * Q[k][i]
#                     Qn[k][i] = Q[k][i] + learning_rate * e_ui * P[u][k]
#     # for u in range(1, U):
#     #     for k in range(0, K):
#     #         Pn[u][k] = P[n][k] + learning_rate *
#     return Pn, Qn, (cost / n ) ** 0.5


def optimize(P, Q, learning_rate, train_set):
    Pn = np.array(P, copy=True)
    Qn = np.array(Q, copy=True)

    U, I, K = len(P), len(Q[0]), len(P[0])

    cost = 0
    n = 0

    e_ui_map = {}

    for u in range(1, U):

        e_ui_map[u] = {}

        for i in range(1, I):
            if u in train_set.keys() and i in train_set[u].keys():
                e_ui = train_set[u][i] - np.dot(P[u, :], Q[:, i])

                e_ui_map[u][i] = e_ui

                # print(P[u, :])
                # print(Q[:, i])
                # print(e_ui)

                cost += e_ui**2
                n += 1
                Pn[u, :] = Pn[u, :] + learning_rate * e_ui * Q[:, i]
                Qn[:, i] = Qn[:, i] + learning_rate * e_ui * P[u, :]
                # for k in range(0, K):
                #     Pn[u][k] = Pn[u][k] + learning_rate * e_ui * Q[k][i]
                #     Qn[k][i] = Qn[k][i] + learning_rate * e_ui * P[u][k]
    # for u in range(1, U):
    #     for k in range(0, K):
    #         t = 0
    #         for i in range(1, I):
    #             if u in e_ui_map.keys() and i in e_ui_map[u].keys():
    #                 t += e_ui_map[u][i]*Q[k][i]
    #         Pn[u][k] = P[u][k] + learning_rate * t
    #
    # for i in range(1, I):
    #     for k in range(0, K):
    #         t = 0
    #         for u in range(1, U):
    #             if i in e_ui_map.keys() and u in e_ui_map[i].keys():
    #                 t += e_ui_map[u][i]*P[u][k]
    #         Qn[k][i] = Q[k][i] + learning_rate * t
    return Pn, Qn, (cost / n) ** 0.5


def train(P, Q, train_set, iterations=1, learning_rate=0.1):
    for i in range(iterations):
        print("iteration %i" % i)
        P, Q, cost = optimize(P, Q, learning_rate, train_set)
        print("Cost after iteration %i: %f" % (i, cost))
    return P, Q


def evaluate(P, Q, val_set):
    R = np.dot(P, Q)
    cost = 0
    n = 0
    for u in val_set.keys():
        for i in val_set[u].keys():
            cost += (abs(val_set[u][i]-R[u][i]))**2
            # print(val_set[u][i], R[u][i ])
            n += 1
    cost /= n
    cost = cost ** 0.5
    # accuracy = 1-accuracy
    print("Error on Val_Set : %f" % cost)


if __name__ == '__main__':
    P, Q, train_set, test_set, val_set = init_P_Q_set(K=4)
    P, Q = train(P, Q, train_set, iterations=30, learning_rate=0.0002)
    evaluate(P, Q, val_set)

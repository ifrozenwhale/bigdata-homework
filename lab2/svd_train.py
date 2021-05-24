import re
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(filename):
    df = pd.read_csv(filename, encoding='utf8')
    return df


def user_mapping(raw_data, filename):
    print(raw_data)
    user_map = {}
    idx = 0
    for uid in raw_data:
        if uid not in user_map:
            user_map[uid] = idx
            idx += 1
    file = open(f"./data/{filename}.txt", mode='w', encoding='utf8')
    for k, v in user_map.items():
        file.write(str(k)+' '+str(v)+'\n')


def get_map(filename):
    dict_temp = {}
    # 打开文本文件
    file = open(f'./data/{filename}.txt', 'r')

    for line in file.readlines():
        line = line.strip()
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dict_temp[int(k)] = int(v)

    file.close()
    return dict_temp


def data_mapping(filename):
    usermap = get_map('usermap')
    moviemap = get_map('moviemap')
    df = load_data("./data/ratings.csv")
    df['uid'] = df['uid'].apply(lambda x: usermap[x])
    df['mid'] = df['mid'].apply(lambda x: moviemap[x])
    del df['timestamp']
    df.to_csv(filename, index=False)


def build_mat(filename, savaname):
    df = pd.read_csv(filename)
    mat = np.zeros((df['uid'].max(), df['mid'].max()))
    print(mat.shape)
    for index, data in df.iterrows():
        mat[data['uid']-1, data['mid']-1] = data['rating']
        if index % 10000 == 0:
            print(f'index {index}')
    np.save(f"./data/{savaname}_mat", mat)


def load_mat(typename):
    mat = np.load(f"./data/{typename}_mat.npy")
    return mat


def split_dataframe(data, rate):
    split_index = np.cumsum(rate).tolist()[:-1]
    data = data.sample(frac=1)
    splits = np.split(data, [round(x * len(data)) for x in split_index])
    for i in range(len(rate)):
        splits[i]["split_index"] = i
    return splits


def split_dataset():
    df = load_data("./data/ratings.csv")
    user_group = df.groupby(by="uid")
    splits = []
    rate = [0.8, 0.1, 0.1]
    # print(user_group.get_group(1).head())
    for name, group in user_group:
        group_splits = split_dataframe(user_group.get_group(name), rate)
        concat_group_splits = pd.concat(group_splits)
        splits.append(concat_group_splits)

    splits_all = pd.concat(splits)
    splits_list = [splits_all[splits_all["split_index"] == x].drop(
        "split_index", axis=1) for x in range(len(rate))]
    # savefile
    splits_list[0].to_csv("./data/train_ratings.csv", index=False)
    splits_list[1].to_csv("./data/val_ratings.csv", index=False)
    splits_list[2].to_csv("./data/test_ratings.csv", index=False)

    # print(splits_list)


def optimize(train_mat, P, Q, eta):
    # calculate
    P_old = P.copy()
    Q_old = Q.copy()
    M, K, N = P.shape[0], P.shape[1], Q.shape[1]
    valid_cnt = 0
    SSE = 0
    for u in range(M):
        for i in range(N):
            rui_std = train_mat[u, i]
            if rui_std == 0:
                continue
            valid_cnt += 1
            rui_p = np.dot(P[u, :], Q[:, i])
            eui = rui_std - rui_p
            P[u, :] = P[u, :] + eta * eui * Q_old[:, i]
            Q[:, i] = Q[:, i] + eta * eui * P_old[u, :]
            SSE += eui ** 2
    return (SSE / valid_cnt) ** 0.5


def train(train_mat, val_mat,  epochs=30):
    # init p and q
    # init k = 10
    print(val_mat)
    M, N = train_mat.shape[0], train_mat.shape[1]
    iter_list = list(range(1, epochs+1))
    kmin, kmax = 14, 15
    K_list = list(range(kmin, kmax))
    eta_list = [0.01]
    train_cost_dict = {}

    # DEBUG
    P = None
    Q = None
    # K_list = [6]
    # eta_list = [0.005]
    for K in K_list:
        for eta in eta_list:
            print("-"*50)
            print("K: {:5}, eta: {}".format(K, eta))
            P = np.random.randint(1, 3, size=(M, K)) / (K ** 0.5)
            Q = np.random.randint(1, 3, size=(K, N)) / (K ** 0.5)
            train_cost_dict[(K, eta)] = []
            for epoch in range(epochs):
                SSE = optimize(train_mat, P, Q, eta)
                val_SSE = calculate_cost(val_mat, P, Q)
                print("epoch: {:5}, RMSE on training set: {:10}".format(
                    1+epoch, SSE))
                print("epoch: {:5}, RMSE on validation set : {:10}".format(
                    1+epoch, val_SSE))
                train_cost_dict[(K, eta)].append(val_SSE)

    np.save(f"./data/P_mat", P)
    np.save(f"./data/Q_mat", Q)

    plt.figure(figsize=(10, 7))
    k_marks = ['o', '*', '', '+']
    # color_list = ['red', 'blue', 'green', 'dark']
    for (K, eta), sse in train_cost_dict.items():
        plt.plot(iter_list, sse, marker=k_marks[(K-kmin) % 4],
                 label='$K$={} | $\eta=${:.5f}'.format(K, eta))
    # def r_ui
    plt.legend()
    plt.savefig("k_eta_sse_4.pdf")
    plt.show()


def calculate_cost(val_mat, P, Q):
    M, K, N = P.shape[0], P.shape[1], Q.shape[1]
    valid_cnt = 0
    SSE = 0
    for u in range(M):
        for i in range(N):
            rui_std = val_mat[u, i]
            if rui_std == 0:
                continue
            valid_cnt += 1
            rui_p = np.dot(P[u, :], Q[:, i])
            eui = rui_std - rui_p
            SSE += eui ** 2
    if valid_cnt == 0:
        return 0
    return (SSE / valid_cnt) ** 0.5


def plot_detail():
    epochs = 30
    fr = open("./result.txt", encoding='utf8')
    raw_data = fr.read()
    raw_data = raw_data.split(
        '--------------------------------------------------')

    k_dict = {}
    rmse_list = []
    for line in raw_data:
        if len(line) < 1:
            continue
        line = line.split("\n")
        line = [e.strip() for e in line if e != '']
        head = re.sub(r'[\s\t]', '', line[0])
        k = int(head[2:head.index(',')])
        k_dict[k] = []

        for data in line[1:]:
            data = re.sub(r'\s', '', data)
            if 'validation' not in data:
                continue
            idx = data.index('set:')
            rmse = float(data[idx+4:])
            k_dict[k].append(rmse)
        # print('-'*100)
    iter_list = list(range(1, epochs+1))
    kmin, kmax = 6, 15
    K_list = list(range(kmin, kmax))
    plt.figure(figsize=(10, 7))
    k_marks = ['o', '*', '', '+']
    # color_list = ['red', 'blue', 'green', 'dark']

    for K, sse in k_dict.items():
        print(len(iter_list), len(sse))
        plt.plot(iter_list, sse, marker=k_marks[(K-kmin) % 4],
                 label='$K$={}'.format(K))
    # def r_ui
    plt.legend()
    plt.savefig("k_eta_sse_3.pdf")
    plt.show()


def get_topK_index(a, k):
    ind = np.argpartition(a, -k)[-k:]
    return ind


def test(test_mat):
    P = load_mat('P')
    Q = load_mat('Q')
    cost = calculate_cost(test_mat, P, Q)
    print("cost on test set", cost)


def calc_precisionK(P, Q, test_mat, k=5):
    A = np.dot(P, Q)
    M, K, N = P.shape[0], P.shape[1], Q.shape[1]
    judge_list = []
    bingo_list = []
    valid_list = []
    for u in range(M):
        tmp = test_mat[u, :]
        valid_idx = np.argwhere(tmp > 0).ravel()
        invalid_idx = np.argwhere(tmp == 0).ravel()
        valid_tmp = tmp[tmp > 0]
        if len(valid_tmp) < k:
            continue
        top_k_idx_test = get_topK_index(tmp, k)
        tmp_A = A[u, :].copy()
        top_k_idx_A = get_topK_index(tmp_A, k)
        judge = top_k_idx_A == top_k_idx_test
        bingo = judge[judge == True]
        bingo_list.append(len(bingo))
        valid_list.append(len(valid_tmp))
        acc = len(bingo) / len(judge)
        judge_list.append(acc)
    # mean = np.average(judge_list)
    precisionK = np.sum(bingo_list) / (len(bingo_list)*k)
    recallK = np.sum(bingo_list) / np.sum(valid_list)
    print("[precision@K[{:2}] {:.4f}".format(k, precisionK))
    print("[recall@K   [{:2}] {:.4f}".format(k, recallK))
    return precisionK, recallK, len(valid_list)


def draw_precision(test_mat, filename='pr@K.pdf'):
    # test
    P = load_mat("P")
    Q = load_mat("Q")
    p_list = []
    r_list = []
    k_list = range(1, 50)
    ac_len_list = []
    for k in k_list:
        p, r, ac_len = calc_precisionK(P, Q, test_mat, k=k)
        p_list.append(p)
        r_list.append(r)
        ac_len_list.append(ac_len)

    f, ax1 = plt.subplots()
    ax1.plot(k_list, p_list, '*-', color='steelblue', label="precision@K")
    ax1.set_ylabel('precision@K')
    ax1.set_xlabel('K')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('recall@K')
    ax2.plot(k_list, r_list, 'o-', color='#B22222', label='recall@K')
    ax2.legend(loc=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    return p_list, r_list, ac_len_list

    # test(test_mat, P, Q)


def draw_F1(P, Q, len_list, filename="F1@K.pdf"):
    F1 = [2 * p * r / (p + r) for p, r in zip(P, R)]
    f, ax1 = plt.subplots()
    k_list = range(0, len(F1))
    ax1.plot(k_list, F1, '-o', color='steelblue', label="recall@K")
    ax1.set_ylabel('F1@K')
    ax1.set_xlabel('K')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('valid data size')
    ax2.plot(k_list, len_list, '*', color='#B22222', label='data size')
    ax2.legend(loc=2)
    plt.savefig(filename)
    plt.show()
    return F1


def calc_classic_average(train_mat):
    P = load_mat("P")
    Q = load_mat("Q")
    A = np.dot(P, Q)
    B = np.zeros(A.shape)
    for v in range(train_mat.shape[1]):
        # train_mat[u, :]
        movies = train_mat[:, v]
        mean = np.average(movies)
        B[:, v] = mean
    return B


def draw_comp_F1(F1, F1_c):
    f, ax1 = plt.subplots()
    k_list = range(0, len(F1))
    ax1.plot(k_list, F1, '-', color='steelblue', label="svd F1@K")
    ax1.set_ylabel('svd F1@K')
    ax1.set_xlabel('K')
    ax1.legend(loc=1)
    ax2 = ax1.twinx()
    ax2.set_ylabel('classic F1@K')
    ax2.plot(k_list, F1_c, '*', color='#B22222', label='average F1@K')
    ax2.legend(loc=2)
    plt.savefig("./svd_classic_F1@K.pdf")
    plt.show()


if __name__ == '__main__':
    # data = load_data('./data/ratings.csv')
    # user_mapping(data['uid'], 'usermap')
    # user_mapping(data['mid'], 'moviemap')
    # usermap = get_map('usermap')
    # moviemap = get_map('moviemap')
    # data_mapping("./data/simple-rating.csv")

    # split dataset
    split_dataset()

    # build mat
    # build_mat("./data/train_ratings.csv", 'train')
    build_mat("./data/test_ratings.csv", 'test')
    build_mat("./data/val_ratings.csv", 'val')

    # load mat

    train_mat = load_mat("train")
    val_mat = load_mat("val")
    test_mat = load_mat("test")

    # train
    # train(train_mat, val_mat, epochs=30)
    # plot
    # plot_detail()

    # P, R, validsize = draw_precision(test_mat, filename='debug.pdf')
    # F1 = draw_F1(P, R, validsize, filename='f1@debug.pdf')

    # B = calc_classic_average(train_mat)
    # P, R, validsize = draw_precision(B, filename='debug.pdf')
    # F1_c = draw_F1(P, R, validsize, filename='f1@debug.pdf')
    # draw_comp_F1(F1, F1_c)

    test(test_mat)
    test(val_mat)

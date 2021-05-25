from gensim import corpora, models
import jieba
import pandas as pd
import numpy as np
import csv
import random
import scipy.stats as stats
from collections import Counter

import json


def calculate_tf(df):
    tf = {}
    for line in df.itertuples():
        data = eval(line[3])
        # 对于每一个文档 计算 tf[word, j]
        counter = Counter(data)

        for t in counter.most_common():
            if t[0] not in tf:
                tf[t[0]] = {}
            tf[t[0]][line[0]] = t[1] / len(data)
    np.save('data/tf.npy', tf)


def calculate_idf(df):
    idf = {}
    word_cnt = {}
    D = 0
    for c in df['review']:
        D += len(c)
    for line in df.itertuples():
        # print(line[3])
        data = eval(line[3])
        # 对于每一个文档 计算 tf[word, j]
        counter = Counter(data)
        for t in counter.most_common():

            word_cnt[t[0]] = word_cnt.get(t[0], 0) + t[1]
    for word, cnt in word_cnt.items():
        idf[word] = np.log(D/(1+cnt))
    # print(idf)
    np.save('data/idf.npy', idf)


def calculate_tfidf():
    tf = np.load('data/tf.npy', allow_pickle=True).item()
    idf = np.load('data/idf.npy', allow_pickle=True).item()
    tfidf = {}
    for i, t in tf.items():
        tfidf[i] = {}
        for j in t:
            tfidf[i][j] = tf[i][j] * idf[i]
    print(sorted(tfidf, reverse=True))
    np.save('data/tfidf.npy', tfidf)

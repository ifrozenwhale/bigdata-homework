
from gensim import corpora, models
import jieba
import pandas as pd
import numpy as np
import csv
import random
import scipy.stats as stats
from lda import *


def my_LDA(beta=2, alpha=np.array([0.4, 5, 10])):
    """
    （1）	从全局的泊松分布参数为 的分布中生成一个文档的长度 ， 表示 topic-word密度， 越高，主题包含的单词更多，反之包含的单词更少；
    """
    n = 1
    N = stats.poisson.pmf(n, beta)
    print("N={}".format(N))

    """
    （2）	从全局的狄利克雷参数为 的分布中生成当前文档的主题分布 ， 表示 document-topic密度， 越高，文档包含的主题更多，反之包含的主题更少；
    """
    # theta = stats.dirichlet.pdf(np.array([1]), alpha)
    theta = stats.dirichlet.rvs(alpha=alpha, size=1)
    print("theta = {}".format(theta))


def LDA(df, k=20):
    df['review'] = df['review'].apply(lambda x: eval(x))
    num_top = len(df.groupby('cat')['cat'].unique())

    dictionary = corpora.Dictionary(df['review'])
    corpus = [dictionary.doc2bow(words) for words in df['review']]
    lda = models.ldamulticore.LdaMulticore(
        corpus=corpus, id2word=dictionary, num_topics=k, workers=3)
    for topic in lda.print_topics(num_words=8):
        print(topic)
    lda.save('lda.model')


def train_LDA(df, k=20):
    LDA(df, k)


def LDA_to_mat(df, k=20):
    # topic_mapper = ['酒店', '衣服', '']
    lda = models.ldamodel.LdaModel.load('lda.model')
    print(df['review'])
    df['review'] = df['review'].apply(lambda x: eval(str(x)))
    model = models.LdaModel.load('lda.model')
    dictionary = corpora.Dictionary(df['review'])
    corpus = [dictionary.doc2bow(words) for words in df['review']]
    topics_test = (lda.get_document_topics(corpus))

    mat = np.zeros((len(corpus), k))
    print(mat.shape)
    print(len(topics_test))
    for t in range(0, len(corpus)):
        # print("review {} topic {}".format(t, topics_test[t]))
        try:
            for i in range(k):
                mat[t, i] = topics_test[t][i][1]
        except IndexError:
            pass
    np.save("./data/LDA_mat", mat)
    return mat

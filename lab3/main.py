from sklearn.neural_network import MLPClassifier
from sklearn import linear_model, datasets
from gensim import corpora, models
import jieba
import pandas as pd
import numpy as np
import csv
import random
import scipy.stats as stats
from lda import train_LDA, LDA_to_mat
from tfidf import calculate_tf, calculate_idf, calculate_tfidf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm


def load_data(filename):
    df = pd.read_csv(filename)
    return df


extra_stop_words = ['￥', '￣', 'ｓ', 'ｒ', 'ｐ', 'ｏ', 'ｎ', 'ｋ', 'ｉ', 'ｈ', 'ｅ', 'ｄ', 'ａ', '｀', '＾', '＼', 'Ｙ', 'Ｘ', 'Ｗ', 'Ｖ', 'Ｕ', 'Ｔ', 'Ｓ', 'Ｒ', 'Ｐ', 'Ｏ', 'Ｎ', 'Ｍ', 'Ｌ', 'Ｋ', 'Ｊ', 'Ｉ', 'Ｈ', 'Ｇ', 'Ｅ', 'Ｄ', 'Ｃ', 'Ｂ', '＠', '９', '８', '７', '６', '５', '＄', '\ue7a6', '\ue406', '\ue403', '\ue25c', '\ue108',
                    '４', '３', '２', '１', '０', '＂', '\ufeff', '﹡', '﹗', '﹐', '﹏', '︿', '︶', '\ue844', '\ue5e6', '\ue5e5', '\ue4c7', '\ue416', '\ue415', '\ue411', '\ue408', '\ue404', '\ue345', '\ue22e', '\ue219', '\ue139', '\ue056', '\ue023', '\ue019', '\ue00e']


def split_dataframe(data, rate):
    split_index = np.cumsum(rate).tolist()[:-1]
    data = data.sample(frac=1)
    splits = np.split(data, [round(x * len(data)) for x in split_index])
    for i in range(len(rate)):
        splits[i]["split_index"] = i
    return splits


def split_dataset(rate, filename):
    df = load_data(filename)
    cat_group = df.groupby(by="cat")
    splits = []

    # print(user_group.get_group(1).head())
    for name, group in cat_group:
        group_splits = split_dataframe(cat_group.get_group(name), rate)
        concat_group_splits = pd.concat(group_splits)
        splits.append(concat_group_splits)

    splits_all = pd.concat(splits)
    splits_list = [splits_all[splits_all["split_index"] == x].drop(
        "split_index", axis=1) for x in range(len(rate))]
    # savefile
    splits_list[0].to_csv("./data/train.csv", index=False)
    splits_list[1].to_csv("./data/test.csv", index=False)

    # print(splits_list)


def pre_process(df):
    """
    文本预处理的任务是去除文本的噪声信息，例如HTML标签，文本格式转换，检测语句边界等
    最终将文本处理为结构化的数据。本实验选用的数据集已被预处理过
    """
    pass


def split_words(df, savepath):
    """
    由于中文词语没有空格分割，需要首先使用中文分词器为文本分词，并去除停用词。停用词是指对文本内容影响较弱的词汇，需要自行设置停用词表，然后按表将训练文本中的停用词剔除。目前常用jieba分词器进行中文分词。
    """
    fw = open("./utils/stop_words.txt", encoding='utf8')
    stop_words = set(fw.read().split('\n'))
    df['review'] = df['review'].apply(lambda x: list(
        filter(lambda t: t not in stop_words | set(extra_stop_words), jieba.lcut(str(x)))))
    if savepath:
        df.to_csv(savepath, index=False)
    return df


def build_tfidf_mat():
    dic = np.load('./data/tfidf.npy', allow_pickle=True).item()
    df = load_data('./data/split_words.csv')
    M = len(df)
    N = len(dic)

    mat = np.zeros((M, N))
    print(mat.shape)


def evaluate(y_pred, y_test):
    # if gbm:
    # y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    # else:
    # y_pred = gbm.predict(x_test)
    # y_predict = np.argmax(y_pred, axis=1)  # 获得最大概率对应的标签
    y_predict = y_pred

    label_all = ['negative', 'positive']
    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    df = pd.DataFrame(confusion_mat, columns=label_all)
    df.index = label_all
    print('准确率：', metrics.accuracy_score(y_test, y_predict))
    print(df)
    TP = confusion_mat[1, 1]
    FP = confusion_mat[0, 1]
    FN = confusion_mat[1, 0]

    P = TP / (TP + FP)
    R = TP / (TP + FN)
    print("Precision@ {:10}\nRecall@ {:10}".format(P, R))
    print("F1@ {:10}".format(2*P*R/(P+R)))


def build_model():
    df = load_data("./data/split_words.csv")
    x_w = tfidf_transform(df['review'])
    x_train, x_test, y_train, y_test = train_test_split(
        x_w, df['label'], test_size=0.4)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2)

    build_light_gbm(x_train, y_train, x_test, y_test, x_val, y_val)


def build_light_gbm(x_train, y_train, x_test, y_test, x_val=None, y_val=None):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_val = lgb.Dataset(x_val, y_val, reference=lgb_train)
    params = {'max_depth': 5, 'min_data_in_leaf': 20, 'num_leaves': 30,
              'learning_rate': 0.12, 'lambda_l1': 0.1, 'lambda_l2': 0.2,
              'objective': 'binary', 'verbose': -1}

    num_boost_round = 2000
    gbm = lgb.train(params, lgb_train, num_boost_round,
                    verbose_eval=100, valid_sets=lgb_val)
    # gbm = lgb.train(params, lgb_train, num_boost_round)
    gbm.save_model('data/lightGBM_model')
    print('On Train set')
    # evaluate(gbm, x_train_weight, y_train)
    y_pred = gbm.predict(x_train, num_iteration=gbm.best_iteration)
    y_pred = np.round(y_pred)
    print(y_pred)

    evaluate(y_pred, y_train)
    print("On Validation set")
    y_pred = gbm.predict(x_val, num_iteration=gbm.best_iteration)
    y_pred = np.round(y_pred)
    evaluate(y_pred, y_val)

    print("On Test set")
    y_pred = gbm.predict(x_test, num_iteration=gbm.best_iteration)
    y_pred = np.round(y_pred)
    evaluate(y_pred, y_test)


def tfidf_transform(X):
    vectorizer = CountVectorizer(max_features=5000)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(
        vectorizer.fit_transform(X))
    x_w = tf_idf.toarray()
    return x_w


def test_model():
    df = load_data("./data/split_words_test.csv")
    gbm = lgb.Booster(model_file="./data/lightGBM_model")
    x_w = tfidf_transform(df['review'])
    print("On Test set")
    evaluate(gbm, x_w, df['label'])


def build_lda(train=False):
    words_df = load_data("./data/split_words.csv")
    if train:
        train_LDA(words_df, 10)
        LDA_to_mat(words_df, 10)

    mat = np.load("./data/LDA_mat.npy")
    x_train, x_test, y_train, y_test = train_test_split(
        mat, words_df['label'], test_size=0.4)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.3)
    build_light_gbm(x_train, y_train, x_test, y_test, x_val, y_val)
    build_liner(x_train, y_train, x_test, y_test)
    build_network(x_train, y_train, x_test, y_test)


def build_liner(x_train, y_train, x_test, y_test):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    print("\nLiner LogisticRegression:")
    evaluate(y_pred, y_test)


def build_network(x_train, y_train, x_test, y_test):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5,), random_state=1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)  # 预测
    print("\nNeural_network:")
    evaluate(y_pred, y_test)


if __name__ == '__main__':
    raw_filename = "./data/online_shopping_10_cats.csv"
    df = load_data(raw_filename)
    # 1. 划分单词
    # split_words(df, "./data/split_words.csv")

    # 2. LDA

    # train_LDA(words_df)

    # 3. 计算 tfidf
    # calculate_tf(df)
    # calculate_idf(df)
    # calculate_tfidf()

    # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

    # build_model()

    build_lda(train=False)

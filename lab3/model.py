import jieba
import pandas as pd
import numpy as np
import csv
import random


def load_data(filename):
    df = pd.read_csv(filename)
    return df


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


def split_words(df):
    """
    由于中文词语没有空格分割，需要首先使用中文分词器为文本分词，并去除停用词。停用词是指对文本内容影响较弱的词汇，需要自行设置停用词表，然后按表将训练文本中的停用词剔除。目前常用jieba分词器进行中文分词。
    """
    fw = open("./utils/stop_words.txt", encoding='utf8')
    stop_words = fw.read().split('\n')
    df['review'] = df['review'].apply(lambda x: list(
        filter(lambda t: t not in stop_words, jieba.lcut(str(x)))))


if __name__ == '__main__':
    raw_filename = "./data/online_shopping_10_cats.csv"
    train_filename = "./data/train.csv"
    test_filename = "./data/test.csv"
    rate = [0.6, 0.4]

    split_dataset(rate, raw_filename)
    train_df = load_data(train_filename)
    test_df = load_data(test_filename)
    split_words(train_df.head(100))

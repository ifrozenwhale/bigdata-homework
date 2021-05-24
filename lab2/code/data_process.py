import numpy as np
import pandas as pd

import csv


def load_data(filename):
    data = pd.read_csv(filename, header=None, encoding='utf-8',
                       delimiter="::", quoting=csv.QUOTE_NONE, engine='python')
    data.columns = ['uid', 'mid', 'rating', 'timestamp']
    data.to_csv("./data/ratings.csv", index=False)


if __name__ == '__main__':
    load_data("./data/ratings.dat")

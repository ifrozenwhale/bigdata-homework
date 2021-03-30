from scipy.stats import pearsonr
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import palettable


def load_data(filepath='./iris-data/iris.csv'):
    df = pd.read_csv(filepath)
    return df


def draw_hist(data, savepath="./len.png"):
    n, bins, patches = plt.hist(
        x=data, label='sepal length(cm)',  alpha=0.5, edgecolor="black")
    plt.xticks(bins)  # x轴刻度设置为箱子边界

    for patch in patches:  # 每个箱子随机设置颜色
        patch.set_facecolor(random.choice(
            palettable.colorbrewer.qualitative.Dark2_7.mpl_colors))

    print("频数", n)  # 频数
    print("边界", bins)  # 箱子边界
    print("数目", len(patches))  # 箱子数

    # 直方图绘制分布曲线
    plt.plot(bins[:10], n, '--', color='#2ca02c')
    plt.legend()
    plt.savefig(savepath, dpi=600)
    plt.show()


def draw_box(data, datatype=0, savepath="./iris-box-length"):
    data = iris_data['sepal.length'] if datatype == 0 else iris_data
    sns.boxplot(data=data, width=0.3)
    plt.savefig(savepath, dpi=600)
    plt.show()


def draw_different_box(data, attribute='sepal.length', savepath="./iris-box-diff.png"):
    sns.boxplot(data=data, width=0.3, x='variety', y=attribute)
    plt.savefig(savepath, dpi=600)
    plt.show()


def five_number(nums):
    # 五数概括 Minimum（最小值）、Q1、Median（中位数、）、Q3、Maximum（最大值）
    Minimum = min(nums)
    Maximum = max(nums)
    Q1 = np.percentile(nums, 25)
    Median = np.median(nums)
    Q3 = np.percentile(nums, 75)

    IQR = Q3-Q1
    lower_limit = Q1-1.5*IQR  # 下限值
    upper_limit = Q3+1.5*IQR  # 上限值

    return Minimum, Q1, Median, Q3, Maximum, lower_limit, upper_limit


def draw_pairs_plot(data, savepath="./iris-pair-plot.png"):
    graph = sns.pairplot(data)
    print(graph)
    rp_dict = {}

    def corrfunc(x, y, **kws):

        (r, p) = pearsonr(x, y)
        rp_dict[x.name+'-'+y.name] = (round(r, 3), round(p, 3))
        ax = plt.gca()
        ax.annotate("r = {:.2f} ".format(r),
                    xy=(.1, .9), xycoords=ax.transAxes)
        ax.annotate("p = {:.3f}".format(p),
                    xy=(.4, .9), xycoords=ax.transAxes)

    graph.map(corrfunc)

    # plt.savefig(savepath, dpi=600)
    # plt.show()
    for item in rp_dict.items():
        print("{:<30}  {:<10}".format(item[0], str(item[1])))


def draw_qq_plot_help(data, xname, yname, ax):
    ls1 = sorted(data[xname])
    ls2 = sorted(data[yname])
    sns.regplot(x=ls2, y=ls1, ci=None, color='steelblue',
                line_kws={'color': 'r'}, ax=ax, label=xname+'-'+yname)
    ax.legend()


def draw_qq_plot(data, savepath):
    # plt.style.use("seaborn-white")
    n = len(data.columns)-1
    fig, axes = plt.subplots(n, n, figsize=[15, 15])
    labels = list(data.columns)[:-1]
    for i in range(n):
        for j in range(n):
            if j <= i:
                axes[j, i].axis('off')
            else:
                draw_qq_plot_help(data, labels[i], labels[j], ax=axes[j, i])
    plt.savefig(savepath, dpi=600)
    plt.show()


if __name__ == '__main__':
    plt.style.use("seaborn-whitegrid")
    iris_data = load_data(filepath="./iris-data/iris.csv")
    length_data = iris_data['sepal.length']
    # draw_hist(length_data, savepath="./iris_hist_len.png")
    # draw_box(iris_data)
    # minx, q1, mid, q3, maxv, _, _ = five_number(length_data)
    # print(f"Five number: ({minx}, {q1}, {mid}, {q3}, {maxv})")
    # draw_different_box(iris_data, attribute='sepal.length')
    # draw_pairs_plot(iris_data, './iris-pair-plot-pearson.png')
    draw_qq_plot(iris_data, "./iris-qq-test.png")

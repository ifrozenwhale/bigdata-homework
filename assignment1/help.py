import matplotlib.pyplot as plt
import collections
import numpy as np
x = [13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25,
     25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70]


print(np.mean(x))
print(np.median(x))
dic = collections.Counter(x)
print(dic)
print(np.max(x), np.min(x))
print((np.max(x) + np.min(x)) / 2)
# 25%
print(np.percentile(x, 25))
# 75%分位数
print(np.percentile(x, 75))


def fiveNumber(nums):
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


print(fiveNumber(x))


data = [9.5, 26.5,	7.8,	17.8,	31.4,	25.9,	27.4,	27.2,	31.2,
        34.6,	42.5,	28.8,	33.4,	30.2	, 34.1	, 32.9,	41.2	, 35.7]
x = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
y = [9.5, 26.5,	7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2,
     34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
plt.boxplot(x=y)
plt.tick_params(top='off', right='off')
plt.savefig('./box4.png', bbox_inches='tight', dpi=600)
plt.show()

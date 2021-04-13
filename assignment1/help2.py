
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

x = [23, 23, 27, 27, 39, 41, 47, 49, 50, 52, 54, 54, 56, 57, 58, 58, 60, 61]
y = [9.5, 26.5,	7.8, 17.8, 31.4, 25.9, 27.4, 27.2, 31.2,
     34.6, 42.5, 28.8, 33.4, 30.2, 34.1, 32.9, 41.2, 35.7]
print(np.mean(x), np.mean(y))
print(np.median(x), np.median(y))
print(np.std(x), np.std(y))
plt.style.use('seaborn-white')
plt.xlabel("age")
plt.ylabel("Body fat rate")
plt.scatter(x, y)
plt.savefig("./scatter.png", dpi=600)
plt.show()

ls1 = sorted(x)
ls2 = sorted(y)
sns.regplot(x=ls2, y=ls1, ci=None, color='steelblue', line_kws={'color': 'r'})
plt.savefig("./q-q.png", dpi=600)
plt.show()

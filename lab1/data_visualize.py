import palettable
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决无法显示符号的问题
sns.set(font='SimHei', font_scale=0.8)        # 解决Seaborn中文显示问题


def load_data(filename):
    df = pd.read_csv(filename)
    return df


def draw_area_type(df):
    # 按照地区、职务、工作类别、领域、公司年龄等字段制作合适的图表（饼图、直方图等），能够直观有重点地展示出基础数据的分布情况并得出适当结论。
    # 地区分布
    area_ser = df['地区'].value_counts()
    thehold = 30
    other_area_ser = area_ser[area_ser < thehold]
    area_ser = area_ser[area_ser > thehold]
    other_area_sum = other_area_ser.sum()
    tmp_data = pd.Series([other_area_sum], index=['其他'])
    area_ser = area_ser.append(tmp_data)

    plt.figure(figsize=(20, 20))
    area_ser.plot.pie()
    plt.savefig("./img/area_pie.pdf")
    # plt.show()
    plt.close()


def draw_title_type(df):
    # 职务
    title_ser = df['职务等级'].value_counts()
    title_ser.plot.pie()
    plt.title("职务分布饼图")
    plt.savefig("./img/title_pie.pdf")
    plt.close()


def draw_work_type(df):
    work_ser = df['工作类别'].value_counts()
    work_ser.plot.bar()
    plt.title("工作类别条形图")
    plt.savefig("./img/work_hist.pdf")
    plt.close()


def draw_company_type(df):
    work_ser = df['公司类别'].value_counts()
    work_ser.plot.bar()
    plt.title("公司领域条形图")
    plt.savefig("./img/company.pdf", bbox_inches='tight')
    plt.close()


def draw_company_year(df):
    df['公司年龄'] = df['公司年龄'].apply(
        lambda x: 0 if x == '不足一年' else int(x[:-1]) if isinstance(x, str) else -1)
    work_ser = sorted(df['公司年龄'].values.tolist())
    work_ser = [e for e in work_ser if e > -1]

    n, bins, patches = plt.hist(
        x=work_ser, alpha=0.5, edgecolor="black")
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

    plt.title("公司年龄直方图")
    plt.savefig("./img/company_year.pdf", bbox_inches='tight')
    plt.close()


# 用户是否认证与公司是否认证之间的关联关系
def draw_ca_relation(df):
    user_ca = df['是否认证'].values.tolist()
    comp_ca = df['公司是否认证'].values.tolist()
    data = [0, 0, 0, 0]
    # 00, 10, 01, 11
    for u, c in zip(user_ca, comp_ca):
        if u == 0 and c == 0:
            data[0] += 1
        elif u > c:
            data[1] += 1
        elif u < c:
            data[2] += 1
        else:
            data[3] += 1

    plt.figure(figsize=(8, 8))

    sns.set(font='SimHei', font_scale=1.8,
            style="whitegrid")        # 解决Seaborn中文显示问题

    sns.scatterplot(x=[0, 1, 0, 1], y=[0, 0, 1, 1],
                    hue=data, size=data, sizes=(100, 800), palette='BuGn_d')
    plt.xticks([-0.5, 0, 1, 1.6], ['', '未认证（个人）', '认证（个人）', ''])
    plt.yticks([-0.5, 0, 1, 1.6],  ['', '未认证（企业）',
                                    '认证（企业）', ''], rotation=90, verticalalignment='center')
    # plt.xlabel("个人")
    # plt.ylabel("企业")
    plt.legend(loc="right")
    plt.savefig("./img/ca_relation.pdf")
    plt.show()
    plt.close()


def draw_company_money_year(df):
    # df['公司年龄'] = df['公司年龄'].apply(
    #     lambda x: 0 if x == '不足一年' else int(x[:-1]) if isinstance(x, str) else -1)
    tmp_df = df[(df['注册资金'] > 0) & (df['注册资金'] < 1e11)]
    comp_year = tmp_df['公司年龄']

    comp_money = tmp_df['注册资金']

    comp = [(xi, yi) for xi, yi in zip(comp_year, comp_money)]
    sorted_Xy = sorted(comp)
    sorted_X = [xi for xi, _ in sorted_Xy]
    sorted_y = [yi for _, yi in sorted_Xy]

    sns.scatterplot(sorted_X, sorted_y)
    plt.savefig("./img/comp_money_year.pdf")
    # plt.show()
    plt.close()

    group_df = tmp_df.groupby(by='注册资金等级')


def get_q4_limit(data):
    p = np.percentile(data, [0, 25, 50, 75, 100])
    IQR = p[3] - p[1]
    up = p[3] + IQR*1.5
    down = p[1] - IQR*1.5
    return up, down


def get_dif_ca_title(df):

    ca_df = df[df['是否认证'] == 1]
    no_df = df[df['是否认证'] == 0]
    ca_counts = ca_df['职务等级'].value_counts()
    no_counts = no_df['职务等级'].value_counts()
    ca_sum = np.sum(ca_counts)
    no_sum = np.sum(no_counts)
    ca_counts = ca_counts.apply(lambda x: x/ca_sum)
    no_counts = no_counts.apply(lambda x: x/no_sum)
    no_counts.sort_index(inplace=True)
    ca_counts.sort_index(inplace=True)
    xlen = 5
    x = np.arange(0, 0+xlen, 1)
    width = xlen / 15
    x2 = x + width
    plt.bar(x, ca_counts.values, width=width, label='认证人群', alpha=0.5)
    plt.bar(x2, no_counts.values, width=width, label='非认证人群',
            alpha=0.5, tick_label=no_counts.index)

    plt.xlabel("职务等级")
    plt.ylabel("在对应人群中的比例")
    plt.title("认证/非认证人群中职务等级分布的差异对比")
    plt.legend()
    plt.savepdf("./img/ca_title_relation.pdf")
    plt.show()
    plt.close()


def get_dif_compca_age(df):

    tmp_df['公司是否认证'] = tmp_df['公司是否认证'].apply(
        lambda x: '是' if x == 1 else '否')

    sns.boxplot(data=tmp_df, x='公司是否认证', y='公司年龄')
    plt.title("公司是否认证与公司年龄关系图")
    plt.savefig("./img/company_ca_age.pdf")
    plt.show()
    plt.close()
    # ca_df = df[df['公司是否认证'] == 1]
    # no_df = df[df['公司是否认证'] == 0]
    # ca_counts = ca_df['注册资金'].
    # no_counts = no_df['注册资金'].value_counts()


def get_dif_compca_money(df):
    tmp_df = df[(df['注册资金'] > 0) & (df['注册资金'] < 1e10)]
    tmp_df['公司是否认证'] = tmp_df['公司是否认证'].apply(
        lambda x: '是' if x == 1 else '否')

    sns.boxplot(data=tmp_df, x='公司是否认证', y='注册资金')
    plt.title("公司是否认证与公司注册资金关系图")
    plt.savefig("./img/company_ca_money.pdf")
    plt.show()
    plt.close()


def get_dif_compca_money_level(df):

    sns.set(font='SimHei', font_scale=0.5)        # 解决Seaborn中文显示问题
    fig, axes = plt.subplots(1, 2)  # 创建画布
    ca_df = df[df['公司是否认证'] == 1]
    no_df = df[df['公司是否认证'] == 0]
    ca_data = ca_df['注册资金等级'].value_counts()
    no_data = no_df['注册资金等级'].value_counts()
    labels = no_data.index
    # colors = ['peru', 'steelblue']  # 每块对应的颜色
    explode = tuple(0.03 for i in range(len(labels)))  # 将每一块分割出来，值越大分割出的间隙越大
    axes[0].pie(ca_data,
                # colors='Blues',
                labels=labels,
                explode=explode,
                autopct='%.2f%%',  # 数值设置为保留固定小数位的百分数
                shadow=False,  # 无阴影设置
                startangle=90,  # 逆时针起始角度设置
                pctdistance=0.5,  # 数值距圆心半径背书距离
                labeldistance=1.05  # 图例距圆心半径倍距离
                )  # 在axes[0]上绘制男性饼图
    axes[0].axis('equal')  # x,y轴刻度一致，保证饼图为圆形
    axes[0].legend(loc='best')
    axes[0].set_title('认证公司注册资金等级')

    axes[1].pie(no_data,
                # colors='Blues',
                labels=labels,
                explode=explode,
                autopct='%.2f%%',  # 数值设置为保留固定小数位的百分数
                shadow=False,  # 无阴影设置
                startangle=180,  # 逆时针起始角度设置
                pctdistance=0.5,  # 数值距圆心半径背书距离
                labeldistance=1.05  # 图例距圆心半径倍距离
                )  # 在axes[1]上绘制女性饼图
    axes[1].axis('equal')  # x,y轴刻度一致，保证饼图为圆形
    axes[1].set_title('非认证公司注册资金等级')
    axes[1].legend(loc='best')
    plt.savefig("./img/company_ca_money_level.pdf")
    plt.show()


if __name__ == '__main__':
    df = load_data("./general_data.csv")
    # draw_area_type(df)
    # draw_title_type(df)
    # draw_work_type(df)
    # draw_company_type(df)
    # draw_company_year(df)
    # draw_ca_relation(df)
    # draw_company_money_year(df)
    # get_dif_ca_title(df)
    # get_dif_compca_money(df)
    get_dif_compca_money_level(df)
# 六角图

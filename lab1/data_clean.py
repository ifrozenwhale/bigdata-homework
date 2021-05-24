import numpy as np
import pandas as pd
import re


def load_data(filename):
    df = pd.read_csv(filename, encoding='utf8', dtype={'tel': object})

    return df


def valid_number(x):
    b = re.match('^1[358]\d{9}$|^147\d{8}$|^179\d{8}$', str(x))
    return b is not None


def data_fix(df):
    """找出不符合规则的数据并进行修正
    字段内部多余的空格
    字段为空或特殊字符的数据
    字段“tel”不符合电话格式的数据等

    选择合适的处理手段，例如删除、置默认值、置平均值/众数等，将数据处理为规则的数据
    """

    # 查看数据集情况
    print(df.info())

    # 去除字段内部的多余的空格（连续多个空格）
    df = df.applymap(lambda x: re.sub("\s+", " ", str(x))
                     if not pd.isnull(x) else x)
    # 去除字段内首尾空格
    df = df.applymap(lambda x: x.strip() if type(x) is str else x)

    # 检查字段为空的数据
    # 定义空值需要删除的字段
    non_null_attrs = ['是否认证', '姓名', '部门-职务', '公司名称']
    df = df.dropna(subset=non_null_attrs)

    # 定义空值需要进行填充默认值的字段
    null_default_attrs = ['工作内容', 'qq',
                          'email', 'tel', '注册资金',  '教育经历', '工作经历']
    df[null_default_attrs] = df[null_default_attrs].fillna('/')

    # 定义需要用出现最频繁的值填充的字段
    most_values_attrs = ['公司是否认证', '公司成立时间', '公司年龄', ]
    # 按照公司名称进行分组，这三个字段与公司有关
    df[most_values_attrs] = df.groupby('公司名称')[most_values_attrs].apply(
        lambda x: x.fillna(x.dropna().mode()))

    # 检查手机号码
    # df[df['tel'] != '/']['tel'].to_csv("./test.csv")

    df['tel'].apply(lambda x: x if valid_number(x) else '/')
    df.to_csv('./data_fix.csv', index=False)
    return df


def data_format(df):
    # print(df['部门-职务'])
    df['部门-职务'] = df['部门-职务'].astype(str)
    df = df.drop(df[df['部门-职务'].map(len) < 2].index)
    df[['部门', '职务']] = df['部门-职务'].str.split(' ', 1, expand=True)
    print(df[['部门', '职务']])

    def money_format(x):
        if '万' in x:
            return float(x[:-1])*1e4
        elif '亿' in x:
            return float(x[:-1]) * 1e8
        else:
            return -1

    df['注册资金'] = df['注册资金'].apply(
        lambda x: money_format(x))
    df = df.drop(df[df['注册资金'] == -1].index)

    import time

    def time_format(x):
        if '/' in str(x):
            return time.strftime(
                "%Y-%m-%d", time.strptime(x, "%Y/%m/%d"))
        else:
            return "0000-00-00"
    df['公司成立时间'] = df['公司成立时间'].apply(lambda x: time_format(x))

    df['公司年龄'] = df['公司年龄'].apply(
        lambda x: 0 if x == '不足一年' else int(x[:-1]) if isinstance(x, str) else -1)
    # 公司年龄
    # df['公司年龄']

    # print(df['教育经历'])
    # 中国人民大学 1 市场营销 2017-01-01 2019-01-01 | 西藏大学 0 文秘 2013-01-01 2016-01-01
    edu_df = pd.DataFrame()
    edu_attrs = ['学校', '类型', '专业', '开始时间', '结束时间']
    edu_df[['姓名', '教育经历']] = df[['姓名', '教育经历']]
    edu_df_tmp = edu_df['教育经历'].str.split(
        '|', expand=True).stack().reset_index(level=1, drop=True).rename("教育经历")
    edu_df = edu_df.drop('教育经历', axis=1).join(edu_df_tmp)

    edu_df["教育经历"] = edu_df['教育经历'].astype(str)
    edu_df[edu_attrs] = edu_df["教育经历"].str.strip().str.split(' ',
                                                             4, expand=True)

    del edu_df['教育经历']
    edu_df.to_csv("./edu_info.csv", index=False)

    # print(df['工作经历'])
    # 广州计测检测技术有限公司 副总监 2017-04-15 现在 | 天祥集团 客户经理 2004-08-15
    work_df = pd.DataFrame()
    work_attrs = ['企业', '职位', '部门', '开始时间', '结束时间']
    work_df[['姓名', '工作经历']] = df[['姓名', '工作经历']]

    work_df_tmp = work_df['工作经历'].str.split(
        '|', expand=True).stack().reset_index(level=1, drop=True).rename('工作经历')
    work_df = work_df.drop('工作经历', axis=1).join(work_df_tmp)
    print(work_df)
    work_df["工作经历"] = work_df['工作经历'].astype(str)

    work_df[work_attrs] = work_df["工作经历"].str.strip().str.split(' ',
                                                                4, expand=True)
    del work_df['工作经历']
    work_df.to_csv("./work_info.csv", index=False)

    del df['工作经历']
    del df['教育经历']
    del df['部门-职务']

    df['职务'] = df['职务'].fillna(df['部门'])

    df.to_csv("./basic_info.csv", index=False)
    return df


def data_generalization(df):
    def check_item(key, target_title_list):
        for title in target_title_list:
            if title in str(key):
                return True
        return False
    # 职务等级划分为A级（关键词“董事/主席”等）
    # B级（关键词“总经理/总裁/副总经理”等）
    # C级（关键词“总监/副总监/经理/副经理/主任/主管”等）
    # D级（关键词“工程师/xx员/实习生”等）
    # E级（其他）
    level_A_list = ['董事', '主席']
    level_B_list = ['总经理', '总裁', '副总经理']
    level_C_list = ['总监', '副总监', '经理', '副经理', '主任', '主管']
    level_D_list = ['工程师', '员', '实习生']

    def title_trans(x):

        if check_item(x, level_A_list):
            return 'A'
        elif check_item(x, level_B_list):
            return 'B'
        elif check_item(x, level_C_list):
            return 'C'
        elif check_item(x, level_D_list):
            return 'D'
        else:
            return 'E'

    df['职务等级'] = df['职务'].apply(lambda x: title_trans(x))
    # 工作类别划分为市场类（关键词“销售/市场/客户”等）
    # 技术类（关键词“业务/技术/项目”等）
    # 营销类（关键词“营销/宣传”等）
    # 其他类（关键词“财务/运营/行政/人力”等）

    def department_trans(x):
        def check_work(target_title_list):
            for title in target_title_list:
                if title in str(x):
                    return True
            return False
        market_list = ['销售', '市场', '客户']
        tech_list = ['业务', '技术', '项目']
        ad_list = ['营销', '宣传']

        if check_item(x, market_list):
            return '市场类'
        elif check_item(x, tech_list):
            return '技术类'
        elif check_item(x, ad_list):
            return '营销类'
        else:
            return '其他类'

    df['工作类别'] = df['部门'].apply(lambda x: department_trans(x))

    # 公司按照地区和领域进行划分
    # 领域为科技类（关键词“科技/软件/信息技术”等）
    # 文化、传媒广告类（关键词“文化/传媒/广告”等）
    # 咨询类（关键词“咨询”等）
    # 管理类（关键词“管理”等）
    # 贸易类（关键词“贸易/商贸/科贸/工贸”等）
    # 其他类（关键词“机械/设备/建筑”等）
    def company_trans(x):
        tech_list = ['科技', '软件', '信息技术']
        culture_list = ['文化', '传媒', '广告']
        adv_list = ['咨询']
        ma_list = ['管理']
        trading_list = ['贸易', '商贸', '科贸', '工贸']

        if check_item(x, tech_list):
            return '科技类'
        elif check_item(x, culture_list):
            return '文化、传媒广告类'
        elif check_item(x, adv_list):
            return '咨询类'
        elif check_item(x, ma_list):
            return '管理类'
        elif check_item(x, trading_list):
            return '贸易类'
        else:
            return '其他类'

    df['公司类别'] = df['公司名称'].apply(lambda x: company_trans(x))

    # 注册资金等级划分为
    # “1000万以下”
    # “1000万以上5000万以下”
    # “5000万以上1亿以下”
    # “1亿以上”
    def money_trans(x):
        if x < 1e7:
            return '1000万以下'
        elif x < 5e7:
            return '1000万以上5000万以下'
        elif x < 1e8:
            return '5000万以上1亿以下'
        else:
            return '1亿以上'

    df['注册资金等级'] = df['注册资金'].apply(lambda x: money_trans(float(x)))

    df.to_csv("./general_data.csv", index=False)
    return df


def get_city_names(filename="./city.txt"):
    f = open(filename, encoding='utf8')
    data = f.readlines()
    data = [d.strip().split(" ")[1] for d in data]
    return data


def create_city(df):
    city_names = get_city_names()

    def area_trans(x):
        for name in city_names:
            if name[:-1] in x and len(name) > 2:
                return name
        return '未知'

    df['地区'] = df['公司名称'].apply(lambda x: area_trans(x))
    df.to_csv("./general_data.csv", index=False)


if __name__ == '__main__':
    df = load_data("./general_data.csv")
    print(df.info())
    # df = load_data("./general_data.csv")
    # df = load_data("./data_fix.csv")
    # df = data_fix(df)
    # df = data_format(df)
    # df = data_generalization(df)
    # create_city(df)

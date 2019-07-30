import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)

def read_data(path, dtype_dict):
    '''
    读取数据文文件
    :param path: 文件路径
    :return:  数据框
    '''
    return pd.read_csv(path,  dtype=dtype_dict)

def process(df):
    '''
    预处理
    :param data: 要处理的数据框
    :return: 处理后的数据框
    '''

    for col in df.columns:
        if df.loc[:, col].dtype == 'object':
            del df[col]
        else:
            df.loc[:, col] = df[col].fillna(0)  # 将float和int类型缺失值填充为0
            if col in ['1月末逾期天数', '2月末逾期天数', '3月末逾期天数', '4月末逾期天数', '5月末逾期天数', '6月末逾期天数']:
                df.loc[:, f'{col}_逾期周期'] = pd.cut(df[col], bins=[-1, 0, 29, 59,  89,  119, 149, 179, 99999],
                                                      labels=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])  # 离散化
                del df[col]
    # df.to_csv('/home/kdd/Desktop/df.csv')

    return df


def table_transform(df):
    '''
    将处理后的数据框转换为特定表格
    :param df: 处理后的数据框
    :return: 目标表格
    '''

    df_period_month = pd.DataFrame()
    for i in range(1, 7):
        df_period_month.insert(i-1, column= f'{i}月', value=df.groupby(by=[f'{i}月末逾期天数_逾期周期'])[f'{i}月末应收余额'].sum())
    df_period_month.index.name = '周期'
    # df_period_month.to_csv('/home/kdd/Desktop/月度各周期应收款.csv')  # 输出月度各周期应收款

    return df_period_month


def mobility(df):
    '''
    计算迁移率
    :param df: 月度各期应收款
    :return: 迁移率
    '''
    mobility_df = pd.DataFrame(columns=['2月', '3月', '4月', '5月', '6月'],
                               index=['C0~C1', 'C1~C2', 'C2~C3', 'C3~C4', 'C4~C5', 'C5~C6', 'C6~C7'])
    # 计算各月迁移率
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            if i > 0 and j>0:
                mobility_df.iloc[i-1, j-1] = df.iloc[i, j] / df.iloc[i-1, j-1]
    # 计算月平均迁移率
    for i in mobility_df.index:
        mobility_df.loc[i, '月平均迁移率'] = mobility_df.loc[i, :].mean()

    return mobility_df


def net_loss_rate(df, recovery_rate_avg, rr_C7):
    '''
    计算净损失率
    :param df: 迁移率
    :return: 周期净损失率
    '''
    net_loss_rate_df = pd.DataFrame(columns=['逾期天数', '净损失率'],
                                    index=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
    # C0~C6净损失率
    for i in range(len(df.index)):
        net_loss_rate_df.loc[net_loss_rate_df.index[i], '净损失率'] = df.iloc[i:, -1].prod() * (1 - recovery_rate_avg)
    # C7净损失率
    net_loss_rate_df.loc['C7', '净损失率'] = 1 - rr_C7
    # net_loss_rate_df.to_csv('/home/kdd/Desktop/月度各周期净损失率.csv')  # 输出月度各周期净损失率

    return net_loss_rate_df


def provision(net_loss_rate_df, df_period_month):
    '''
    计算各周期拨备额及拨备率
    :param net_loss_rate_df: 周期净损失率
    :param df_period_month: 月度周期应收款
    :return: 各周期拨备额及拨备率
    '''
    provision_df = pd.concat([net_loss_rate_df, df_period_month.iloc[:, -1]], axis=1)
    provision_df.loc[:, '拨备额'] = provision_df.loc[:, '6月末应收余额'] * provision_df.loc[:, '净损失率']
    provision_rate = provision_df.loc[:, '拨备额'].sum() / provision_df.loc[:, '6月末应收余额'].sum()
    # provision_df.to_csv('/home/kdd/Desktop/各周期拨备额.csv')  # 输出各周期拨备额

    return provision_rate


# 设置字段类型
raw_data_dtype = {'借据号': str, '合同开始日期': str, '合同结束日期': str, '1月末应收余额': float, '1月末逾期天数': float,
                  '1月末五级分类': str, '1月末录入日期': str, '2月末应收余额': float, '2月末逾期天数': float, '2月末五级分类': str,
                  '2月末录入日期': str, '3月末应收余额': float, '3月末逾期天数': float, '3月末五级分类': str, '3月末录入日期': str,
                  '4月末应收余额': float, '4月末逾期天数': float, '4月末五级分类': str, '4月末录入日期': str, '5月末应收余额': float,
                  '5月末逾期天数': float, '5月末五级分类': str, '5月末录入日期': str, '6月末应收余额': float, '6月末逾期天数': float,
                  '6月末五级分类': str, '6月末录入日期': str}




if __name__ == '__main__':

    # 1、读取贷款明细数据
    raw_data = read_data(path='/home/kdd/python/DATA/bs1000.csv', dtype_dict=raw_data_dtype)
    # print(raw_data.head(3))

    # 2、预处理：缺失值和离散化
    df = process(raw_data)

    # 3、将贷款明细透视为月度各周期透支额
    df_period_month = table_transform(df)

    # 4、计算迁移率
    mobility_df = mobility(df_period_month)

    # 5、计算净损失率
    net_loss_rate = net_loss_rate(mobility_df, 0.2, 0.21)
    #
    # # 6、计算拨备额和拨备率
    # provision = provision(net_loss_rate_df=net_loss_rate, df_period_month=df_period_month)
    print(mobility_df)





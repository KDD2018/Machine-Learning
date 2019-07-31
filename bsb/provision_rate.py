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

def process(df, period_dict):
    '''
    预处理
    :param data: 要处理的数据框
    :period_dict: 周期字典
    :return: 处理后的数据框
    '''
    period_month = pd.DataFrame()
    for i in range(1,7):
        i = str(i)
        # print(i)
        cols = [col for col in df.columns if i in col]
        df_i = df.loc[:, cols].dropna()
        df_i.loc[:, '逾期周期'] = pd.cut(df[f'{i}月末逾期天数'], bins=[-1, 0, 30, 60,  90,  120, 180, 99999],
                                     labels=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])  # 逾期天数离散为逾期周期
        s = df_i.groupby(by=['逾期周期'])[f'{i}月末应收余额'].sum()  # 按周期对月度应收余额分组统计
        # df_i.groupby(by=[f'{i}月末五级分类'])['逾期周期'].apply(lambda x: print(x))  # 五级分类怎么分？？？
        period_month.insert(int(i)-1, column=f'{i}月末应收余额', value=s)
    # 插入逾期天数说明
    period = pd.DataFrame.from_dict(period_dict, orient='index', columns=['逾期天数'])
    df_period_month = pd.concat([period, period_month], axis=1)
    df_period_month.index.name = '周期'

    return df_period_month


def mobility(df):
    '''
    计算迁移率
    :param df: 月度各期应收款
    :return: 迁移率
    '''
    mobility = pd.DataFrame(columns=['2月', '3月', '4月', '5月', '6月'],
                               index=['C0~C1', 'C1~C2', 'C2~C3', 'C3~C4', 'C4~C5', 'C5~C6'])
    # 计算各月迁移率
    df_ = df.iloc[:, 1:]
    for i in range(len(df_.index)):
        for j in range(len(df_.columns)):
            if i > 0 and j>0:
                mobility.iloc[i-1, j-1] = df_.iloc[i, j] / df_.iloc[i-1, j-1]
    # 计算月平均迁移率
    for i in mobility.index:
        mobility.loc[i, '月平均迁移率'] = mobility.loc[i, :].mean()

    return mobility


def net_loss_rate(df, recovery_rate_avg, rr_C6, period_dict):
    '''
    计算净损失率
    :param df: 迁移率
    :param recovery_rate_avg: 应收款平均回收率
    :param rr_C6: 180+的回收率
    :param period_dict: 周期字典
    :return: 周期净损失率
    '''
    net_loss_rate = pd.DataFrame(columns=['净损失率'],
                                    index=['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6'])
    # C0~C6净损失率
    for i in range(len(df.index)):
        net_loss_rate.loc[net_loss_rate.index[i], '净损失率'] = df.iloc[i:, -1].prod() * (1 - recovery_rate_avg)
    # C6净损失率
    net_loss_rate.loc['C6', '净损失率'] = 1 - rr_C6
    # 插入逾期天数说明
    period = pd.DataFrame.from_dict(period_dict, orient='index', columns=['逾期天数'])
    net_loss_rate_df = pd.concat([period, net_loss_rate], axis=1)
    net_loss_rate_df.index.name = '周期'
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

    return provision_df, provision_rate


# 设置字段类型
raw_data_dtype = {'借据号': str, '合同开始日期': str, '合同结束日期': str, '1月末应收余额': float, '1月末逾期天数': float,
                  '1月末五级分类': str, '1月末录入日期': str, '2月末应收余额': float, '2月末逾期天数': float, '2月末五级分类': str,
                  '2月末录入日期': str, '3月末应收余额': float, '3月末逾期天数': float, '3月末五级分类': str, '3月末录入日期': str,
                  '4月末应收余额': float, '4月末逾期天数': float, '4月末五级分类': str, '4月末录入日期': str, '5月末应收余额': float,
                  '5月末逾期天数': float, '5月末五级分类': str, '5月末录入日期': str, '6月末应收余额': float, '6月末逾期天数': float,
                  '6月末五级分类': str, '6月末录入日期': str}

# 周期划分字典
period_dict = {'C0': '正常', 'C1': '1~30', 'C2': '31~60', 'C3': '61~90', 'C4': '91~120', 'C5': '121~180', 'C6': '180+'}




if __name__ == '__main__':

    # 1、读取贷款明细数据
    raw_data = read_data(path='/home/kdd/python/DATA/贷款明细-2019年1-6月.csv', dtype_dict=raw_data_dtype)
    # print(raw_data.columns)

    # 2、预处理（缺失值和离散化）并将贷款明细透视为月度各周期透支额
    df_period_month = process(raw_data, period_dict=period_dict)
    print(df_period_month)


    # 3、计算迁移率
    mobility_df = mobility(df_period_month)
    print(mobility_df)

    # 4、计算净损失率
    net_loss_rate = net_loss_rate(mobility_df, 0.2, 0.21, period_dict)
    print(net_loss_rate)

    # 5、计算拨备额和拨备率
    provision_df, provision = provision(net_loss_rate_df=net_loss_rate, df_period_month=df_period_month)
    print(provision_df)
    print(f'拨备率为：{provision}')






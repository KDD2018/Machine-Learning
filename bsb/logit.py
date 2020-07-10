#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.utils import shuffle
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression


parser = argparse.ArgumentParser(description='命令行参数测试')
parser.add_argument('--data-path', type=str, default='')
parser.add_argument('--sheet-name', type=str, default='')
args = parser.parse_args()
data_path = args.data_path
sheet_name = args.sheet_name


def get_df(path, sheet):
    """
    获取数据
    :param path: 数据路径
    :param sheet: sheet名称
    :return: 数据框
    """
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    df = shuffle(df)
    return df


def scaler(df):
    """
    标准化
    :param df: 原特征数据
    :return: 标准化结果
    """
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return df


def lr_model(feature, label):
    """
    Logistic Regression
    :param feature: 特征
    :param label: 标签
    :return: LR模型
    """
    lr_model = LogisticRegression()
    lr_model.fit(feature, label)
    score = lr_model.score(feature, label)
    return lr_model, score



if __name__ == '__main__':

    # 获取数据
    df = get_df(path=data_path, sheet=sheet_name)
    df.columns = ['defaulted', 'grade', 'GDP_acc']
    # 标签
    target = df.pop('defaulted')
    # 归一化
    # df['GDP_acc'] = scaler(np.array(df['GDP_acc']).reshape(-1, 1))

    # one-hot编码
    grade_dummy = pd.get_dummies(df['grade'], prefix='grade')
    df = pd.concat([grade_dummy, pd.DataFrame(df['GDP_acc'])], axis=1)

    # 搭建模型
    lr_model, score = lr_model(df, target)

    # 预测
    df_new = pd.read_csv('./forecast.csv')
    grade_dummy = pd.get_dummies(df_new['grade'], prefix='grade')
    df_pre = pd.concat([grade_dummy, pd.DataFrame(df_new['GDP_acc'])], axis=1)

    prob = lr_model.predict_proba(df_pre)
    print(prob)
    df_new['prob'] = prob[:, 1]
    print(df_new)
    print(lr_model.coef_)
    print(lr_model.intercept_)
    df_new.to_csv('./predict.csv', encoding='utf-8')








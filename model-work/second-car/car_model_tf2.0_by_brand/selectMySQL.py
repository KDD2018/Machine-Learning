#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pymysql
import pandas as pd


class SelectMySQL(object):
    '''
    从MySQL获取数据
    '''
    def __init__(self, host, user, passwd, db):
        self.host = host
        self.user = user
        self.passwd = passwd
        self.db = db

    def get_df(self,sql):
        '''
        连接MySQL并返回数据
        :param sql: 查询语句
        :return: 查询结果
        '''
        conn = pymysql.connect(host=self.host,
                               user=self.user,
                               passwd=self.passwd,
                               db=self.db,
                               port=3306,
                               charset='utf8',
                               cursorclass=pymysql.cursors.SSDictCursor,
                               connect_timeout=7200)
        cur = conn.cursor()
        cur.execute(sql)
        data = cur.fetchall()

        # for rec in alldata:
        #     result.append(rec)  # 注意，我这里只是把查询出来的第一列数据保存到结果中了,如果是多列的话，稍微修改下就ok了

        cur.close()
        conn.close()

        return data


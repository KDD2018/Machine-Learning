#coding=utf-8

import requests
import json
import csv
import time
import os
import pandas


def getHtml(typeid,page):
    '''
    根据页数、类型id爬取数据
    :param typeid:类型id
    :param page:第几页
    :return:爬取到的结果
    '''
    url = 'https://api.m.jd.com/api?appid=paimai-search-soa' \
          '&functionId=paimai_unifiedSearch&body=' \
          '{%22apiType%22:2,%22page%22:'+ str(page) + ',%22pageSize%22:40,%22reqSource%22:0,%22paimaiStatus%22:%222%22,%22childrenCateId%22:%22' \
          + typeid + \
          '%22}&loginType=3'
    headers = {
        'referer': 'https://auction.jd.com/sifa_list.html?paimaiStatus=1&childrenCateId=12728',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.125 Safari/537.36'
    }
    ret = requests.get(url,headers=headers)
    print(ret.text)
    return ret.text

def parseHtml(typename,pagedata):
    '''
    提取数据
    :param typename:类型名称
    :param pagedata:每一页的数据
    :return:提取结果列表
    '''
    mydataList = []
    try:
        dataList = json.loads(pagedata)['datas']
        for data in dataList:
            id = data['id']
            try:
                # 省份
                addr1 = data['province']
            except:
                addr1 = ''
            try:
                # 城市名
                addr2 = data['city']
            except:
                addr2 = ''
            try:
                # 标题
                title = data['title']
            except:
                title = ''
            try:
                # 市场价
                shichangjia = data['marketPrice']
            except:
                shichangjia = ''
            try:
                # 评估价
                pinggujia = data['assessmentPrice']
            except:
                pinggujia = ''
            try:
                # 当前价
                dangqianjia = data['currentPrice']
            except:
                dangqianjia = ''
            try:
                # 结束时间
                tmp_endtime = data['endTime']
                endtime = time.strftime('%Y-%m-%d',time.localtime(int(tmp_endtime)/1000))
            except:
                endtime = ''
            try:
                # 几次出价
                bidcount = data['bidCount']
                if bidcount == 0:
                    status = '流拍'
                else:
                    status = '成功'
            except:
                status = ''
            try:
                # 成交价格
                success_price = data['currentPrice']
            except:
                success_price = ''
            mydataList.append([id,typename,addr1,addr2,title,shichangjia,pinggujia,dangqianjia,endtime,success_price,status])
    except Exception as e:
        print(e)
    return mydataList

def saveData(dataList):
    '''
    保存数据到csv文件
    :param dataList:一页的数据
    :return:
    '''
    f = open('data.csv','a',newline='',encoding='utf-8-sig')
    c = csv.writer(f)
    for data in dataList:
        # 逐行写入数据
        c.writerow(data)
    f.close()


if __name__ == '__main__':
    typeNameList = ['住宅用房','商业用房','工业用房','其他用房','土地']
    typeIDList = ['12728', '13809', '13817', '13810', '12730']
    pageList = [3305,1758,75,732,141]
    if not os.path.exists('data.csv'):
        f = open('data.csv','w',newline='',encoding='utf-8-sig')
        c= csv.writer(f)
        c.writerow(['ID','类型','省份','城市','标题','市场价','评估价','当前价','结束时间','成交价','状态'])
        f.close()
    # 遍历页数列表
    for k,pageMax in enumerate(pageList):
        # 遍历每一页
        for page in range(pageMax):
            print(typeNameList[k],page+1,'/',pageMax)
            htmldata = getHtml(typeIDList[k],page+1)
            dataList = parseHtml(typeNameList[k],htmldata)
            saveData(dataList)
            time.sleep(3)

    pd = pandas.read_csv('data.csv')
    pd.to_excel('爬取结果.xlsx',index=None)

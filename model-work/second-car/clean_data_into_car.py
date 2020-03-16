import argparse
import datetime
import logging
import os
import sys
import pymysql
import difflib
import time
import re
import jieba
import traceback
import pymongo

sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utils.mongodb_zlpg_conn import mongo_conn_zlpg

"""
功能：清洗二手车数据
调用方式：--table-name base_data_second_hand_car --db-name valuation_web
作者：魏东
注意：remark有多个状态，1表示已经清洗，2表示未对应上数据（未清洗），3表示异常数据, 4表示车品牌对应不上,5表示车系对应不上，
                    6表示车型对应不上, 7表示报错数据
清洗逻辑：
有两张数据表
1.车类型表 
2.案例表
需要定义两个方法，第一个方法添加车类型表数据，第二个方法添加案例数据，一定要先执行第一个方法，在执行第二个方法。
第一步：往业务表（车类型表）中填数据
    车类型表：主要字段是 name parent_id level
    有name字段，在name字段中添加品牌，车系，车型
    1.先添加品牌数据，pid为0  level 为0
    去重逻辑：添加品牌的数据时，去业务表中去查找是由有这个品牌，有判断重复
    2.添加车系数据 ，pid为对应的品牌的id level为1
    去重逻辑：添加车系的数据时，在业务表中根据品牌的数据筛选出对应的车系，进行比较，如果有，则重复，获取车系对应的id，若没有则添加，并获取id
    3.添加车型数据,pid为对应的车系的id level 为2
    去重逻辑：添加车型数据时，在业务表中，根据pid即车系的id筛选出相关的车型，进行比较，如果有，则重复不添加，若没有则添加
第二步：往案例表中添加数据
    通过title字段，提取出车品牌，车系，车型的数据，没有网站的提取规则不一样，所以需要分开来写。
"""

# 不同网站的数据分开方法清洗，每个网站的数据特别不规则
today = datetime.date.today().strftime("%Y-%m-%d")
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(os.getcwd(), 'clean_data_into_car' + today + '.log'),
                    # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    filemode='a',
                    # 日志格式
                    format='%(process)d %(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] '
                           '%(levelname)s %(message)s')
parser = argparse.ArgumentParser(description='命令行参数测试')
parser.add_argument('--db-name', type=str, default='')
parser.add_argument('--table-name', type=str, default='')
parser.add_argument('--year-month', type=str, default=datetime.datetime.now().strftime('%Y-%m'))
args = parser.parse_args()
db_name = args.db_name
table_name = args.table_name
year_month = args.year_month
logging.info(
    "运行参数db-name:{0},table-name:{1},year-month:{2}".format(db_name, table_name, year_month))


class CleanDataIntoCar:
    def __init__(self):
        self.conn_default = pymysql.connect(host='192.168.0.3', user='pnggu', passwd='Pinggu123',
                                            db='python3', port=3306, charset='utf8',
                                            cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.cursor_default = self.conn_default.cursor()

        self.default_tow = pymysql.connect(host='192.168.0.3', user='clean', passwd='Zlpg1234!',
                                           db=db_name, port=3306, charset='utf8',
                                           cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.cursor_default_tow = self.default_tow.cursor()

        # 车型表
        self.car_type_sql = """SELECT * from base_data_car_type where spider_name='{0}' and remark != '8'"""
        # 二手车案例表
        self.used_car_sell_sql = """SELECT id,content,remark from base_data_second_hand_car where spider_name = '{0}'
        and remark !='1' and remark != '2' and remark != '3' and remark != '4' and remark != '5' and remark != '6'
        and remark != '7'"""
        # self.used_car_sell_sql = """SELECT * from base_data_second_hand_car where spider_name = '{0}'
        #         and  remark = '6'"""

        # 更新remark状态
        self.update_remark_sql = """ update {0} set remark = {1} where id = {2} """
        # 查询车类型 在second_car_types中car_class等于0代表汽车，等于1代表卡车
        self.query_sql = """SELECT name from second_car_types where level = {0} and car_class = 0"""
        # 查询车id
        self.query_car_brand_id_sql = """SELECT id from second_car_types where name = '{0}' and level = {1} and 
        car_class = 0"""
        # 通过车id筛选车系信息 或 通过车系id筛选车型信息
        self.query_car_system_sql = """SELECT id,name from second_car_types where parent_id = {0}"""
        # 案例表去重
        self.repeat_sql = """select IFNULL(count(*),0) as count,id from second_car_sell where price = {price} 
        and meter_mile = {meter_mile} and semiautomatic_gearbox = "{semiautomatic_gearbox}"
        and displacement = {displacement} and emission_standard = "{emission_standard}" 
        and (sell_times = "{sell_times}" or sell_times is NULL) 
        and (year_check_end_time = "{year_check_end_time}" or year_check_end_time is NULL) 
        and (is_have_strong_risk = "{is_have_strong_risk}" or is_have_strong_risk is NULL) 
        and type = "{type}" and brand_id = {brand_id} and vehicle_system_id = {vehicle_system_id} 
        and car_model_id = {car_model_id}"""
        # 插入车型数据
        self.insert_car_type_sql = """insert into second_car_types(createTime,createUser,updateTime,
        updateUser,isPushed,isDeleted,remark,name,level,parent_id) 
            values(%s,"清洗", %s, %s, 0, 0, %s, %s, %s, %s)"""
        # 从新车表中查看car_class
        self.get_car_class = """select car_class, level from new_car_information where car_model_id = {0}"""
        # 插入实例数据
        # 注意：最后的字段level是新添加的字段，之前的type使用level表示，新添加的level用new_level表示
        self.insert_car_sell_sql = """insert into second_car_sell
        (remark,createTime,createUser,updateTime,updateUser,isPushed,isDeleted,source,
        price,register_time,meter_mile,semiautomatic_gearbox,displacement,emission_standard,
        sell_times,year_check_end_time,is_have_strong_risk,business_risk_end_time,img_url,
        risk_27_check,type,detail_url,brand_id,vehicle_system_id,car_model_id,brand_name,
        vehicle_system_name,car_model_name,car_class,model_year, level) 
            values (%s,%s,"清洗", %s, %s, 0, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
        %s,%s,%s,%s,%s,%s,%s)"""

        self.car_class_dict = {'suv': ['大型SUV', '中大型SUV', '小型SUV', '紧凑型SUV', '中型SUV', "SUV"],
                               'saloon': ['小型车', '大型车', '微型车', '中大型车', '中型车', '紧凑型车'],
                               'supercar': '跑车',
                               'pickup': ['皮卡', '微卡', '高端皮卡'], 'mpv': 'MPV', 'minibus': ['轻客', '微面']
                               }
        self.dic_emission_standard = {'1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六',
                                      'VI': '六', 'IV': '四', 'III': '三', 'II': '二', 'I': '一', 'V': '五'}
        self.model_year_dict = {'2020款': 2020, '2019款': 2019, '2018款': 2018, '2017款': 2017, '2016款': 2016,
                                '2015款': 2015, '2021款': 2021, '2022款': 2022, '2023款': 2023, '2024款': 2024,
                                '2014款': 2014, '2013款': 2013, '2012款': 2012, '2011款': 2011, '2010款': 2010, '09款': 2009,
                                '08款': 2008, '07款': 2007, '06款': 2006, '05款': 2005, '04款': 2004, '03款': 2003,
                                '02款': 2002,
                                '01款': 2001, '2000款': 2000, '99款': 1999, '98款': 1998, '97款': 1997, '96款': 1996,
                                '95款': 1995, '94款': 1994, '93款': 1993, '92款': 1992, '91款': 1991}
        # 车品牌统一 "新凯汽车": "奔驰" 中有奔驰牌子
        self.brand_name_dict = {"传祺": "广汽传祺", "吉利": "吉利汽车", "川汽野马": "野马汽车", "长安商用": "长安",
                                "五菱": "五菱汽车", "MG": "名爵", "猎豹": "猎豹汽车", "北京汽车": "北京",
                                "北京现代": "现代", "全新奔腾": "奔腾", "野马": "野马汽车", "比速": "比速汽车", "力帆": "力帆汽车",
                                "莲花": "莲花汽车", "汉腾": "汉腾汽车", "斯威汽车": "SWM斯威汽车", "阿尔法罗密欧": "阿尔法·罗密欧",
                                "华晨金杯": "金杯", "LITE": "ARCFOX", "上汽大通": "上汽大通MAXUS", "理想汽车": "理想智造",
                                "上汽MAXUS": "上汽大通MAXUS", "Polestar极星": "Polestar", "NEVS国能汽车": "国能汽车",
                                "合众汽车": "哪吒汽车", "Karma": "Fisker", "极星": "Polestar", "理想": "理想智造",
                                "宇通": "宇通客车"}
        self.system_name_dict = {"理想ONE": "理想智造ONE", "JCW MINI JCW CLUBMAN": "MINI JCW CLUBMAN",
                                 "JCW MINI JCW COUNTRYMAN": "MINI JCW COUNTRYMAN",
                                 "AC Schnitzer X5": "Schnitzer AC Schnitzer X5",
                                 "AC Schnitzer M3": "Schnitzer AC Schnitzer M3", "幸福e+": "汉腾新能源 幸福e+",
                                 "探歌": "T-ROC探歌", "艾菲": "银隆艾菲", "LAFESTA 菲斯塔": "菲斯塔",
                                 "揽胜运动": "揽胜运动版", "揽胜运动 插电混动": "揽胜运动版新能源",
                                 "马自达3 昂克赛拉": "马自达3 Axela昂克赛拉", "富康ES500": "东风新能源 富康ES500",
                                 "奕泽": "奕泽IZOA", "威旺M35": "北汽威旺M35", "锐骐厢式车": "郑州日产 锐骐厢式车",
                                 "AMG GLC级": "奔驰GLC AMG", "AMG E级": "奔驰E级AMG", "AMG S级": "奔驰S级AMG",
                                 "AMG CLA级": "奔驰CLA级AMG", "AMG C级": "奔驰C级AMG", "AMG GLA级": "奔驰GLA AMG",
                                 "AMG G级": "奔驰G级AMG"}
        # 当前时间
        self.now_time = datetime.datetime.now()

    def re_num(self, value):
        value_list = re.findall(r'\d+\.?\d*', str(value))
        lst = []
        for i in value_list:
            if '.' in i:
                i = float(i)
                lst.append(i)
            else:
                i = int(i)
                lst.append(i)
        return lst

    # 匹配车型规则
    def get_car_model_data(self, car_model_list, car_model_name, model_year):
        # 待清洗的车型数据
        car_model_name = str(car_model_name).replace('版', '型') \
            .replace("4MATIC", "四驱").replace("2MATIC", "二驱").replace('年', '款').replace("前驱", "两驱").replace("AT", "自动") \
            .replace("MT", "手动").replace("（", "(").replace("）", ")")
        car_model_name = self.filter_L_rule(car_model_name)
        for car_model in car_model_list:
            """
            不同的情况：
               众泰T600 2015款 2.0T 自动豪华型
               众泰T600 2015款 2.0T DCT豪华型  DCT就是自动挡，需要统一格式
               别克 英朗 2013款 GT 1.6L 自动时尚版
               2013款 1.6 自动 GT时尚型  顺序也可能不一样，需要相似度匹配
               4MATIC  = 四驱 统一
               
            先通过统一格式方式去进行对应，如果还是对应不上，再通过拆分的方式
            2003款 S 600 和  2006款 S 300  这总情况相似度为1 所以可以先判断是哪年份 
             """
            # 将2019 拼接成 19款
            model_year_ = str(model_year)[2:] + "款"
            # 业务表存在的数据  【L 情况区分对待】
            car_model_name_service = str(car_model.get("name")).replace('版', '型') \
                .replace("4MATIC", "四驱").replace("2MATIC", "二驱").replace("前驱", "两驱").replace("AT", "自动") \
                .replace("MT", "手动").replace("（", "(").replace("）", ")")
            # 还有没有年份的车型，排除在外
            if model_year_ in car_model_name_service or '款' not in car_model_name_service:
                # 过滤L单位
                # 业务表存在的数据
                car_model_name_service = self.filter_L_rule(car_model_name_service).replace('年', '款').replace(' ', '')
                # 待清洗的数据
                car_model_name_case = car_model_name.replace(' ', '')
                # 相似度匹配
                score = difflib.SequenceMatcher(None, car_model_name_service, car_model_name_case).quick_ratio()
                if score == 1:
                    break

        else:
            # 通过拆分的方式去对应车品牌，车系，车型
            for car_model in car_model_list:
                """
                2008款 320i 进取型
                2008款 2.0 自动 320i 进取型 解决这种情况
                """
                car_model_ = car_model.get("name").replace('版', '型').replace("4MATIC", "四驱") \
                    .replace("2MATIC", "二驱").replace("前驱", "两驱").replace("AT", "自动").replace("MT", "手动") \
                    .replace("（", "(").replace("）", ")")
                car_model_ = self.filter_L_rule(car_model_)
                if ' ' in car_model.get("name"):
                    # 过滤L单位之后切割
                    car_model_li = car_model_.split(" ")
                    # 40 TFSI 舒适型(进口)  没有款项，这种情况要过滤
                    if '款' in ''.join(car_model_li) and '款' in car_model_name or '款' not in ''.join(car_model_li) \
                            and '款' not in car_model_name:
                        # 有改款车，需要区分
                        if '改款' in ''.join(car_model_li) and '改款' in car_model_name or '改款' not in ''.join(car_model_li) \
                                and '改款' not in car_model_name:

                            if len(car_model_li) == 2 and car_model_li[0] in car_model_name and car_model_li[1] \
                                    in car_model_name:
                                break

                            elif len(car_model_li) == 3 and car_model_li[0] in car_model_name and car_model_li[1] \
                                    in car_model_name and car_model_li[2] in car_model_name:
                                break

                            elif len(car_model_li) == 4 and car_model_li[0] in car_model_name and car_model_li[
                                1] in car_model_name and \
                                    car_model_li[2] in car_model_name and car_model_li[3] in car_model_name:
                                break

                            elif len(car_model_li) == 5 and car_model_li[0] in car_model_name and car_model_li[1] \
                                    in car_model_name and car_model_li[2] in car_model_name and car_model_li[3] \
                                    in car_model_name \
                                    and car_model_li[4] in car_model_name:
                                break

                            elif len(car_model_li) == 6 and car_model_li[0] in car_model_name and car_model_li[
                                1] in car_model_name and car_model_li[2] in car_model_name and car_model_li[3] \
                                    in car_model_name and car_model_li[4] in car_model_name and car_model_li[5] \
                                    in car_model_name:
                                break

                            elif len(car_model_li) == 7 and car_model_li[0] in car_model_name and car_model_li[
                                1] in car_model_name and car_model_li[2] in car_model_name and car_model_li[
                                3] in car_model_name and car_model_li[4] in car_model_name and car_model_li[5] \
                                    in car_model_name and car_model_li[6] in car_model_name:
                                break
                """
                2018款 300h 行政版 国V
                2018款 300h 行政版
                2015款 改款 C 200 L
                2015款 改款 C 200 L 运动型  通过in的方式匹配车型，要考虑这种情况
                """
                if " " in car_model_name:
                    # 过滤L单位之后切割
                    car_model_name_li = car_model_name.split(' ')
                    if '款' in ''.join(car_model_name_li) and '款' in car_model_ or '款' not in ''.join(car_model_name_li) \
                            and '款' not in car_model_:
                        # 有改款车，需要区分
                        if '改款' in ''.join(car_model_name_li) and '改款' in car_model_ or '改款' not in ''.join(
                                car_model_name_li) \
                                and '改款' not in car_model_:
                            if len(car_model_name_li) == 2 and car_model_name_li[0] in car_model_ \
                                    and car_model_name_li[1] in car_model_:
                                break

                            elif len(car_model_name_li) == 3 and car_model_name_li[0] in car_model_ and \
                                    car_model_name_li[1] in car_model_ and car_model_name_li[2] in car_model_:
                                break

                            elif len(car_model_name_li) == 4 and car_model_name_li[0] in car_model_ and \
                                    car_model_name_li[1] in car_model_ and car_model_name_li[2] \
                                    in car_model_ and car_model_name_li[3] in car_model_:
                                break

                            elif len(car_model_name_li) == 5 and car_model_name_li[0] in car_model_ and \
                                    car_model_name_li[1] in car_model_ and car_model_name_li[2] \
                                    in car_model_ and car_model_name_li[3] in car_model_ \
                                    and car_model_name_li[4] in car_model_:
                                break

                            elif len(car_model_name_li) == 6 and car_model_name_li[0] in car_model_ and \
                                    car_model_name_li[1] in car_model_ and car_model_name_li[2] in car_model_ and \
                                    car_model_name_li[3] in car_model_ and car_model_name_li[4] in car_model_ and \
                                    car_model_name_li[5] in car_model_:
                                break

                            elif len(car_model_name_li) == 7 and car_model_name_li[0] in car_model_ and \
                                    car_model_name_li[1] in car_model_ and car_model_name_li[2] in car_model_ and \
                                    car_model_name_li[3] in car_model_ and car_model_name_li[4] in car_model_ and \
                                    car_model_name_li[5] in car_model_ and car_model_name_li[6] in car_model_:
                                break
            else:
                car_model = {}
        #  确保车型名字和业务表中存的是一致的
        car_model_name = car_model.get("name")
        car_model_id = car_model.get("id")
        return {"car_model_id": car_model_id, 'car_model_name': car_model_name}

    # 过滤L单位规则
    def filter_L_rule(self, car_model_name):
        """
               2013款 1.6L 自动舒适版
               2013款 1.6 自动 舒适版  解决这种情况
               奔驰C级 2019款 C 300 L 运动版
               奔驰C级 2019款 C 300 运动版 这两种车型不一样
        :param car_model_name: 根据L不同表现，过滤L单位
        :return: 过滤单位L之后的数据
        """
        if 'L' in str(car_model_name) and ' L' not in str(car_model_name):
            car_model_name = str(car_model_name).replace('L', '')
        # 车型中也有国几的情况，要统一
        for k, v in self.dic_emission_standard.items():
            if k in car_model_name:
                car_model_name = car_model_name.replace(k, v)
        return car_model_name.upper()

    # 过滤标题规则
    def filter_title_rule(self, title, brand_name, source):
        """
        :param title:  标题
        :param brand_name:  车品牌
        :param source: 数据来源（网站）
        :return:
        """
        if brand_name == "Smart":
            title = title.replace("Smart", "").strip()
        # 瓜子网中的全新胜达就是业务表中的胜达车系名称
        if brand_name == "现代" and "全新胜达" in title:
            title = title.replace("全新胜达", "胜达")
        # 人人车斯威汽车在二手车基本信息表中是SWM斯威汽车
        elif brand_name == "斯威汽车" and "SWM斯威汽车" not in title:
            title = title.replace("斯威汽车", "SWM斯威汽车")
        elif brand_name == "三菱":
            if "劲炫" in title and "进口" in title and "劲炫(进口)" not in title:
                title = title.replace("劲炫", "ASX劲炫(进口)")
            elif "欧蓝德" in title and "进口" in title and "欧蓝德(进口)" not in title:
                title = title.replace("欧蓝德", "欧蓝德(进口)")
            elif "帕杰罗" in title and "进口" in title and "帕杰罗(进口)" not in title:
                title = title.replace("帕杰罗", "帕杰罗(进口)")
            elif "帕杰罗·劲畅" in title and "进口" in title:
                title = title.replace("帕杰罗·劲畅", "帕杰罗·劲畅(进口)")
            elif "太空车" in title and "进口" in title and "太空车（进口）" not in title:
                title = title.replace("太空车", "太空车（进口）")
        elif brand_name == "名爵" and "MG" in title:
            title = title.replace("MG", "名爵")
        # 业务表中存的是马自达3 Axela昂克赛拉，就是瓜子网的昂克赛拉
        elif "马自达 昂克赛拉" in title:
            title = title.replace("马自达 昂克赛拉", "马自达3 Axela昂克赛拉")
        # 铃木 天语 SX4 2013款 1.6L 手动酷锐型
        elif brand_name == "铃木":
            title = title.replace("铃木", "").strip()
            # 吉姆尼修改成吉姆尼(进口)
            if "吉姆尼" in title and "进口" in title and "吉姆尼(进口)" not in title:
                title = title.replace("吉姆尼", "吉姆尼(进口)")
        # 业务表中的车系是凯迪拉克ATS(进口)
        elif brand_name == "凯迪拉克":
            if "凯迪拉克ATS" in title and "进口" in title and "凯迪拉克ATS(进口)" not in title:
                title = title.replace("凯迪拉克ATS", "凯迪拉克ATS(进口)")
            elif "凯迪拉克CTS" in title and "进口" in title and "凯迪拉克CTS(进口)" not in title:
                title = title.replace("凯迪拉克CTS", "凯迪拉克CTS(进口)")
        # 丰田 RAV4 2016款 2.0L CVT两驱风尚版
        elif brand_name == "丰田":
            if "RAV4" in title and "进口" in title and "RAV4（进口）" not in title:
                title = title.replace("RAV4", "RAV4（进口）")
            elif "普拉多" in title and "进口" in title and "普拉多(进口)" not in title:
                title = title.replace("普拉多", "普拉多(进口)")
            elif "汉兰达" in title and "进口" in title and "汉兰达(进口)" not in title:
                title = title.replace("汉兰达", "汉兰达(进口)")
            elif "兰德酷路泽" in title and "进口" in title and "兰德酷路泽(进口)" not in title:
                title = title.replace("兰德酷路泽", "兰德酷路泽(进口)")
            elif "凯美瑞" in title and "进口" in title and "凯美瑞(海外)" not in title:
                title = title.replace("凯美瑞", "凯美瑞(海外)")
        # 瓜子二手车中只有一个车系高尔夫，业务表中区分高尔夫和高尔夫(进口)
        elif brand_name == "大众":
            if "高尔夫" in title and "进口" in title and "高尔夫(进口)" not in title:
                title = title.replace("高尔夫", "高尔夫(进口)")
            elif "迈腾" in title and "进口" in title and "迈腾(进口)" not in title:
                title = title.replace("迈腾", "迈腾(进口)")
            elif "高尔夫新能源" in title and "进口" in title and "高尔夫新能源(进口)" not in title:
                title = title.replace("高尔夫新能源", "高尔夫新能源(进口)")
            elif "R36" in title and "进口" in title and "R36（进口）" not in title:
                title = title.replace("R36", "R36（进口）")
        elif brand_name == "宝马":
            if "宝马5系" in title and "进口" in title and "宝马5系(进口)" not in title:
                title = title.replace("宝马5系", "宝马5系(进口)")
            elif "宝马1系" in title and "进口" in title and "宝马1系(进口)" not in title:
                title = title.replace("宝马1系", "宝马1系(进口)")
            elif "宝马3系" in title and "进口" in title and "宝马3系(进口)" not in title:
                title = title.replace("宝马3系", "宝马3系(进口)")
            elif "宝马X3" in title and "进口" in title and "宝马X3(进口)" not in title:
                title = title.replace("宝马X3", "宝马X3(进口)")
            elif "宝马X1" in title and "进口" in title and "宝马X1(进口)" not in title:
                title = title.replace("宝马X1", "宝马X1(进口)")
            elif "宝马2系旅行车" in title and "进口" in title and "宝马2系旅行车(进口)" not in title:
                title = title.replace("宝马2系旅行车", "宝马2系旅行车(进口)")
        elif brand_name == "奔驰":
            if "威霆" in title and "进口" in title and "威霆(进口)" not in title:
                title = title.replace("威霆", "威霆(进口)")
            elif "奔驰C级" in title and "进口" in title and "奔驰C级(进口)" not in title:
                title = title.replace("奔驰C级", "奔驰C级(进口)")
            elif "奔驰A级" in title and "进口" in title and "奔驰A级(进口)" not in title:
                title = title.replace("奔驰A级", "奔驰A级(进口)")
            elif "唯雅诺" in title and "进口" in title and "唯雅诺(进口)" not in title:
                title = title.replace("唯雅诺", "唯雅诺(进口)")
            elif "奔驰E级" in title and "进口" in title and "奔驰E级(进口)" not in title:
                title = title.replace("奔驰E级", "奔驰E级(进口)")
            elif "奔驰GLA" in title and "进口" in title and "奔驰GLA(进口)" not in title:
                title = title.replace("奔驰GLA", "奔驰GLA(进口)")
            elif "奔驰GLC" in title and "进口" in title and "奔驰GLC(进口)" not in title:
                title = title.replace("奔驰GLC", "奔驰GLC(进口)")
        elif brand_name == "奥迪":
            if "奥迪A3" in title and "进口" in title and "奥迪A3(进口)" not in title:
                title = title.replace("奥迪A3", "奥迪A3(进口)")
            elif "奥迪A4" in title and "进口" in title and "奥迪A4(进口)" not in title:
                title = title.replace("奥迪A4", "奥迪A4(进口)")
            elif "奥迪A6" in title and "进口" in title and "奥迪A6(进口)" not in title:
                title = title.replace("奥迪A6", "奥迪A6(进口)")
            elif "奥迪Q5" in title and "进口" in title and "奥迪Q5(进口)" not in title:
                title = title.replace("奥迪Q5", "奥迪Q5(进口)")
            elif "奥迪A3新能源" in title and "进口" in title and "奥迪A3新能源(进口)" not in title:
                title = title.replace("奥迪A3新能源", "奥迪A3新能源(进口)")
            elif "奥迪Q3" in title and "进口" in title and "奥迪Q3(进口)" not in title:
                title = title.replace("奥迪Q3", "奥迪Q3(进口)")
            elif "R8" in title and "进口" in title and "R8（进口）" not in title:
                title = title.replace("R8", "R8（进口）")
        # 业务表中揽胜极光车系分为 揽胜极光 和 揽胜极光(进口) 在瓜子网中需要区分
        elif brand_name == "路虎":
            if "揽胜极光" in title and "进口" in title and "揽胜极光(进口)" not in title:
                title = title.replace("揽胜极光", "揽胜极光(进口)")
            elif "发现神行" in title and "进口" in title and "发现神行(进口)" not in title:
                title = title.replace("发现神行", "发现神行(进口)")
        elif brand_name == "Jeep":
            if "指南者" in title and "进口" in title and "指南者(进口)" not in title:
                title = title.replace("指南者", "指南者(进口)")
            elif "自由光" in title and "进口" in title and "自由光(进口)" not in title:
                title = title.replace("自由光", "自由光(进口)")
            elif "大切诺基" in title and "进口" in title and "大切诺基(进口)" not in title:
                title = title.replace("大切诺基", "大切诺基(进口)")
        elif brand_name == "本田":
            if "飞度" in title and "进口" in title and "飞度(进口)" not in title:
                title = title.replace("飞度", "飞度(进口)")
        elif brand_name == "标致":
            if "标致206" in title and "进口" in title and "标致206(进口)" not in title:
                title = title.replace("标致206", "标致206(进口)")
            elif "标致307" in title and "进口" in title and "标致307(进口)" not in title:
                title = title.replace("标致307", "标致307(进口)")
            elif "标致207" in title and "进口" in title and "标致207(进口)" not in title:
                title = title.replace("标致207", "标致207(进口)")
            elif "标致308" in title and "进口" in title and "标致308(进口)" not in title:
                title = title.replace("标致308", "标致308(进口)")
            elif "标致3008" in title and "进口" in title and "标致3008(进口)" not in title:
                title = title.replace("标致3008", "标致3008(进口)")
            elif "标致4008" in title and "进口" in title and "标致4008(进口)" not in title:
                title = title.replace("标致4008", "标致4008(进口)")
        elif brand_name == "DS":
            if "DS 5" in title and "进口" in title and "DS 5（进口）" not in title:
                title = title.replace("DS 5", "DS 5（进口）")
        elif brand_name == "福特":
            if "锐界" in title and "进口" in title and "锐界(进口)" not in title:
                title = title.replace("锐界", "锐界(进口)")
            elif "福克斯" in title and "进口" in title and "福克斯(进口)" not in title:
                title = title.replace("福克斯", "福克斯(进口)")
            elif "嘉年华" in title and "进口" in title and "嘉年华(进口)" not in title:
                title = title.replace("嘉年华", "嘉年华(进口)")
        elif brand_name == "克莱斯勒":
            if "克莱斯勒300C" in title and "进口" in title and "克莱斯勒300C(进口)" not in title:
                title = title.replace("克莱斯勒300C", "克莱斯勒300C(进口)")
            elif "大捷龙" in title and "进口" in title and "大捷龙(进口)" not in title:
                title = title.replace("大捷龙", "大捷龙(进口)")
            elif "大捷龙PHEV" in title and "进口" in title and "大捷龙PHEV(进口)" not in title:
                title = title.replace("大捷龙PHEV", "大捷龙PHEV(进口)")
            elif "克莱斯勒君王" in title and "进口" in title and "克莱斯勒君王(进口)" not in title:
                title = title.replace("克莱斯勒君王", "克莱斯勒君王(进口)")
        elif brand_name == "雷诺":
            if "科雷傲" in title and "进口" in title and "科雷傲(进口)" not in title:
                title = title.replace("科雷傲", "科雷傲(进口)")
        elif brand_name == "铃木":
            if "吉姆尼(进口)" in title and "进口" in title and "吉姆尼(进口)" not in title:
                title = title.replace("吉姆尼", "吉姆尼(进口)")
        elif brand_name == "马自达":
            if "马自达8" in title and "进口" in title and "马自达8(进口)" not in title:
                title = title.replace("马自达8", "马自达8(进口)")
            elif "马自达3" in title and "进口" in title and "马自达3(进口)" not in title:
                title = title.replace("马自达3", "马自达3(进口)")
            elif "马自达CX-7" in title and "进口" in title and "马自达CX-7(进口)" not in title:
                title = title.replace("马自达CX-7", "马自达CX-7(进口)")
            elif "马自达CX-5" in title and "进口" in title and "马自达CX-5(进口)" not in title:
                title = title.replace("马自达CX-5", "马自达CX-5(进口)")
        elif brand_name == "讴歌":
            if "讴歌RDX" in title and "进口" in title and "讴歌RDX(进口)" not in title:
                title = title.replace("讴歌RDX", "讴歌RDX(进口)")
        elif brand_name == "起亚":
            if "嘉华" in title and "进口" in title and "嘉华(进口)" not in title:
                title = title.replace("嘉华", "嘉华(进口)")
            # 有起亚K5 和 起亚K5新能源两个车系，要排除
            elif "起亚K5" in title and "进口" in title and "新能源" not in title:
                title = title.replace("起亚K5", "起亚K5新能源")
        elif brand_name == "日产":
            if "奇骏" in title and "进口" in title and "奇骏(进口)" not in title:
                title = title.replace("奇骏", "奇骏(进口)")
            elif "楼兰" in title and "进口" in title and "楼兰(海外)" not in title:
                title = title.replace("楼兰", "楼兰(海外)")
        elif brand_name == "斯柯达":
            if "昊锐" in title and "进口" in title and "昊锐(进口)" not in title:
                title = title.replace("昊锐", "昊锐(进口)")
            elif "Yeti" in title and "进口" in title and "Yeti(进口)" not in title:
                title = title.replace("Yeti", "Yeti(进口)")
            elif "明锐" in title and "进口" in title and "明锐(进口)" not in title:
                title = title.replace("明锐", "明锐(进口)")
            elif "速派" in title and "进口" in title and "速派(进口)" not in title:
                title = title.replace("速派", "速派(进口)")
        elif brand_name == "沃尔沃":
            if "沃尔沃S40" in title and "进口" in title and "新能源" not in title and "沃尔沃S40(进口)" not in title:
                title = title.replace("沃尔沃S40", "沃尔沃S40(进口)")
            elif "沃尔沃XC60" in title and "进口" in title and "新能源" not in title and "沃尔沃XC60(进口)" not in title:
                title = title.replace("沃尔沃XC60", "沃尔沃XC60(进口)")
            elif "沃尔沃S90" in title and "进口" in title and "新能源" not in title and "沃尔沃S90(进口)" not in title:
                title = title.replace("沃尔沃S90", "沃尔沃S90(进口)")
        elif brand_name == "现代":
            if "途胜" in title and "进口" in title and "途胜(进口)" not in title:
                title = title.replace("途胜", "途胜(进口)")
            elif "胜达" in title and "进口" in title and "胜达(进口)" not in title:
                title = title.replace("胜达", "胜达(进口)")
            elif "索纳塔" in title and "进口" in title and "索纳塔(进口)" not in title:
                title = title.replace("索纳塔", "索纳塔(进口)")
        elif brand_name == "雪佛兰":
            if "科帕奇" in title and "进口" in title and "科帕奇(进口)" not in title:
                title = title.replace("科帕奇", "科帕奇(进口)")
        elif brand_name == "雪铁龙":
            if "雪铁龙C4 Aircross" in title and "进口" in title and "雪铁龙C4 Aircross(进口)" not in title:
                title = title.replace("雪铁龙C4 Aircross", "雪铁龙C4 Aircross(进口)")
            elif "雪铁龙C5" in title and "进口" in title and "雪铁龙C5(进口)" not in title:
                title = title.replace("雪铁龙C5", "雪铁龙C5(进口)")
            elif "雪铁龙C6" in title and "进口" in title and "雪铁龙C6(进口)" not in title:
                title = title.replace("雪铁龙C6", "雪铁龙C6(进口)")
        elif brand_name == "英菲尼迪":
            if "英菲尼迪QX50" in title and "进口" in title and "英菲尼迪QX50(进口)" not in title:
                title = title.replace("英菲尼迪QX50", "英菲尼迪QX50(进口)")
        return title

    # 清洗二手车案例信息
    def clean_car_sell_data(self):
        spider_names = ["second_hand_car_renrenche", "second_hand_car_guazi", "second_hand_car_xin",
                        "second_hand_car_autohome", "second_hand_car_taoche"]

        for spider_name in spider_names:
            source = None
            if spider_name == "second_hand_car_xin":
                source = '优信网'
            elif spider_name == "second_hand_car_renrenche":
                source = "人人车"
            elif spider_name == "second_hand_car_autohome":
                source = "汽车之家"
            elif spider_name == "second_hand_car_guazi":
                source = "瓜子网"
            elif spider_name == "second_hand_car_taoche":
                source = "淘车网"
            # monogodb_local = pymongo.MongoClient("mongodb://127.0.0.1:27017/")
            # collection = monogodb_local[spider_name]["全国"]
            collection = mongo_conn_zlpg[spider_name]["全国"]
            # index_info = collection.index_information()
            # if "id_1" in index_info:
            #     collection.drop_index("id_1") .batch_size(2)
            collection.create_index([("id", pymongo.ASCENDING)])
            content_list = collection.find({'$and': [{"is_Processed": {"$ne": 1}},
                                                     {"is_Processed": {"$ne": 2}},
                                                     {"is_Processed": {"$ne": 3}},
                                                     {"is_Processed": {"$ne": 4}},
                                                     {"is_Processed": {"$ne": 5}},
                                                     {"is_Processed": {"$ne": 6}},
                                                     {"is_Processed": {"$ne": 7}}]},
                                           no_cursor_timeout=True).sort([("id", pymongo.ASCENDING)])

            for content in content_list:
                if not content_list:
                    logging.error("没有查询到{0} 的数据".format(content))
                    return
                mog_id = content["id"]
                # self.cursor_default.execute(self.used_car_sell_sql.format(spider_name, ))
                # used_car_datas = self.cursor_default.fetchall()
                # for used_car in used_car_datas:
                #     data_id = used_car.get("id")
                #     content = eval(used_car['content'])
                #     if not content:
                #         logging.error('解析失败')
                #         continue

                # 标题和价格都不能为空
                if not content.get("title") and not content.get("car_price"):
                    self.update_mongodb(3, mog_id, collection)
                    # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                    # self.conn_default.commit()
                    continue

                # 车价格（元）
                price = 0
                if content.get("car_price"):
                    price = str(content.get("car_price")).replace('万', '')
                    if price == "金融特惠" or price == "抢购价":
                        self.update_mongodb(3, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                        # self.conn_default.commit()
                        continue
                    else:
                        price = int(float(price) * 10000)

                # 过滤价格等于0的情况
                if price == 0:
                    self.update_mongodb(3, mog_id, collection)
                    # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                    # self.conn_default.commit()
                    continue

                # 车类型【大型/中型/SUV】
                level = 0
                if content.get("level") and content.get("level").strip() != "-" and content.get("level") != "不限":
                    level = content.get("level").strip().upper()
                    if level == "面包车":
                        level = "微面"
                    elif level == "紧凑型":
                        level = "紧凑型车"
                    elif level == "低端皮卡":
                        level = "皮卡"
                    elif level == "中大型":
                        level = "中大型车"
                    elif level == "小型":
                        level = "小型车"
                    elif level == "中型":
                        level = "中型车"
                    elif level == "微型":
                        level = "微型车"

                # 备注
                remark = None
                if content.get("remark"):
                    remark = content.get("remark")

                # 图片
                img_url = None
                if content.get("img_url"):
                    img_url = content.get("img_url")
                    img_url = str(img_url).replace("[", '').replace(']', '')

                # 默认地址
                detail_url = None
                if content.get("detail_url"):
                    detail_url = content.get("detail_url")

                # 表显里程(里）
                meter_mile = 0
                if content.get("meter_mile"):
                    meter_mile = content.get("meter_mile")
                    if "万" in str(meter_mile):
                        meter_mile_list = self.re_num(meter_mile)
                        if len(meter_mile_list) == 1:
                            meter_mile = int(meter_mile_list[0] * 10000)
                        else:
                            meter_mile = 0

                    elif meter_mile == "百公里内" and "万" not in str(meter_mile):
                        self.update_mongodb(3, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                        # self.conn_default.commit()
                        continue
                if not meter_mile:
                    self.update_mongodb(3, mog_id, collection)
                    continue

                # 过户次数
                sell_times = None
                if content.get("sell_times"):
                    sell_times = str(content.get("sell_times")).replace('次', '').replace('过户', '')
                    if int(sell_times) > 10:
                        self.update_mongodb(3, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                        # self.conn_default.commit()
                        continue

                # 排量(l)
                displacement = 0
                if content.get("displacement") and content.get("displacement") != "无" and content.get(
                        "displacement") != "--":
                    displacement = str(content.get("displacement")).replace("L", '').replace('T', '')
                    if '*' in str(displacement):
                        displacement = 0

                # 上牌时间
                register_time = None
                if content.get("registe_time") and content.get("registe_time") != "未上牌":
                    register_time = content.get("registe_time")
                    if spider_name == "second_hand_car_xin":
                        register_time = str(register_time).replace("年", '-').replace("月", '-').replace("上牌", '01')
                    elif spider_name == "second_hand_car_autohome":
                        if len(str(register_time)) == 6:
                            register_time = str(register_time)[0:4] + "-" + str(register_time)[4:6] + "-01"
                    elif spider_name == "second_hand_car_guazi":
                        if len(str(register_time)) == 7 and "-" in str(register_time):
                            register_time = str(register_time) + "-01"
                    elif spider_name == "second_hand_car_taoche" or spider_name == "second_hand_car_renrenche":
                        register_time = self.manage_register_time(register_time)

                # 有部分上牌时间大于当前的时间，为异常值，需要过滤
                if register_time:
                    register_time_ = datetime.datetime.strptime(register_time, '%Y-%m-%d')
                    if register_time_ > self.now_time:
                        self.update_mongodb(3, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                        # self.conn_default.commit()
                        continue

                # 27项排除重大事故检测
                risk_27_check = 0
                if content.get("risk_27_check"):
                    risk_27_check = content.get("risk_27_check")
                    # 在瓜子网中情况
                    if type(risk_27_check) is list:
                        risk_27_check = ' '.join(risk_27_check).strip().replace(' ', '').replace(',,', ',')
                    else:
                        risk_27_check = str(risk_27_check).strip().replace(' ', '').replace(',,', ',')

                # 汽车之家和淘车网没有这个数据，需要设置为空值
                if spider_name == "second_hand_car_autohome" or spider_name == "second_hand_car_taoche":
                    risk_27_check = None

                # 排放标准
                emission_standard = 0
                if content.get("emission_standard") and content.get("emission_standard") != "-" and content.get(
                        "emission_standard") != "--" and str(content.get("emission_standard")).strip() != "- -" \
                        and content.get("emission_standard") != "无":
                    emission_standard = str(content.get("emission_standard")).strip()
                # 将国1或国I 转换储层 国一
                if emission_standard:
                    for k, v in self.dic_emission_standard.items():
                        if k in emission_standard:
                            emission_standard = emission_standard.replace(k, v)

                # 变速箱
                semiautomatic_gearbox = 0
                if content.get("semiautomatic_gearbox") and content.get("semiautomatic_gearbox") != "- -":
                    semiautomatic_gearbox = str(content.get("semiautomatic_gearbox")).strip()
                    if semiautomatic_gearbox == "自动":
                        semiautomatic_gearbox = "AT自动"
                    elif semiautomatic_gearbox == "手动":
                        semiautomatic_gearbox = "MT手动"
                    elif semiautomatic_gearbox == "双离合":
                        semiautomatic_gearbox = "DCT双离合"
                    elif semiautomatic_gearbox == "半自动":
                        semiautomatic_gearbox = "AMT半自动"

                # 商业险到期时间
                business_risk_end_time = None
                if content.get("business_risk_end_time") and content.get("business_risk_end_time") != "--":
                    business_risk_end_time = content.get("business_risk_end_time")
                    if business_risk_end_time == "已过期":
                        business_risk_end_time = None
                    elif len(str(business_risk_end_time)) == 7 and str(business_risk_end_time)[4] == '-':
                        business_risk_end_time = str(business_risk_end_time) + "-01"
                    # 瓜子网情况
                    elif len(str(business_risk_end_time)) == 6 and str(business_risk_end_time)[4] != '-':
                        business_risk_end_time = str(business_risk_end_time)[0:4] + '-' + str(business_risk_end_time)[
                                                                                          4:6] + '-01'
                    elif len(str(business_risk_end_time)) == 6 and str(business_risk_end_time)[4] == '-':
                        business_risk_end_time = str(business_risk_end_time) + "-01"

                # 交强险时间
                is_have_strong_risk = None
                if content.get("is_have_strong_risk") and content.get("is_have_strong_risk") != "--":
                    is_have_strong_risk = content.get("is_have_strong_risk")
                    if is_have_strong_risk == "已过期":
                        is_have_strong_risk = None
                    elif len(str(is_have_strong_risk)) == 5:
                        is_have_strong_risk = str(is_have_strong_risk)[0:4] + '-0' + str(is_have_strong_risk)[4] + "-01"
                    elif len(str(is_have_strong_risk)) == 6 and str(is_have_strong_risk)[4] != '-':
                        is_have_strong_risk = str(is_have_strong_risk)[0:4] + '-' + str(is_have_strong_risk)[
                                                                                    4:6] + '-01'
                    elif len(str(is_have_strong_risk)) == 6 and str(is_have_strong_risk)[4] == '-':
                        is_have_strong_risk = str(is_have_strong_risk) + "-01"
                    elif len(str(is_have_strong_risk)) == 7 and str(is_have_strong_risk)[4] == '-':
                        is_have_strong_risk = str(is_have_strong_risk) + "-01"
                    elif len(str(is_have_strong_risk)) == 10 and str(is_have_strong_risk)[4] == '-' and \
                            str(is_have_strong_risk)[7] == '-':
                        is_have_strong_risk = content.get("is_have_strong_risk")

                # 年检到期时间
                year_check_end_time = None
                if content.get("year_check_end_time") and content.get("year_check_end_time") != "--" \
                        and content.get("year_check_end_time") != "未知":
                    year_check_end_time = content.get("year_check_end_time")
                    if year_check_end_time == "已过期":
                        year_check_end_time = None
                    elif len(str(year_check_end_time)) == 7 and str(year_check_end_time)[4] == '-':
                        year_check_end_time = str(year_check_end_time) + "-01"
                    elif len(str(year_check_end_time)) == 10 and str(year_check_end_time)[4] == '-' and \
                            str(year_check_end_time)[7] == '-':
                        year_check_end_time = content.get("year_check_end_time")
                    elif len(str(year_check_end_time)) == 5:
                        year_check_end_time = str(year_check_end_time)[0:4] + '-0' + str(year_check_end_time)[
                            4] + "-01"
                    elif len(str(year_check_end_time)) == 6 and str(year_check_end_time)[4] != '-':
                        year_check_end_time = str(year_check_end_time)[0:4] + '-' + str(year_check_end_time)[
                                                                                    4:6] + '-01'
                    elif len(str(year_check_end_time)) == 6 and str(year_check_end_time)[4] == '-':
                        year_check_end_time = str(year_check_end_time) + "-01"
                # 过滤年检到期时间
                if year_check_end_time:
                    flag = self.filter_year_check_end_time(year_check_end_time)
                    if flag:
                        self.update_mongodb(3, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '3', used_car['id']))
                        # self.conn_default.commit()
                        continue

                # 年款 【如2019】
                model_year = None
                for k_year, v_year in self.model_year_dict.items():
                    if k_year in content.get("title"):
                        model_year = v_year
                        break

                isDeleted = 0
                if year_check_end_time == register_time:
                    isDeleted = 1

                # 车品牌
                brand_name = None
                # 车系
                vehicle_system_name = None
                # 另一种获取方式的车系名称
                vehicle_system_name_one = None
                # 车型
                car_model_name = None
                brand_id = None
                vehicle_system_id = None
                car_model_id = None
                vehicle_system_name_other = 0
                if spider_name == "second_hand_car_autohome":
                    if content.get("car_brand"):
                        brand_name = content.get("car_brand")
                        if brand_name == "传祺":
                            brand_name = "广汽传祺"
                    self.cursor_default_tow.execute(self.query_car_brand_id_sql.format(brand_name, 0))
                    brand_id_list = self.cursor_default_tow.fetchall()
                    # 判断车型表中是否存在该品牌，存在获取id
                    if brand_id_list:
                        brand_id = brand_id_list[0].get("id")
                    else:
                        self.update_mongodb(2, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue

                    # 奥迪RS 5 2014款 RS 5 Coupe 特别版
                    # 大众CC 2009款 3.6FSI 顶配版 针对这两种情况，筛选数车型和车系名字
                    title = content.get("title")
                    vehicle_system_model_list = title.split(" ", 2)
                    if len(vehicle_system_model_list) == 3:
                        if '款' in str(vehicle_system_model_list[1]):
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                        elif '款' not in str(vehicle_system_model_list[1]) and '款' in str(vehicle_system_model_list[2]):
                            vehicle_system_name = str(vehicle_system_model_list[0]) + " " + str(
                                vehicle_system_model_list[1])
                            car_model_name = vehicle_system_model_list[2]
                        else:
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                    elif len(vehicle_system_model_list) == 2:
                        vehicle_system_name = vehicle_system_model_list[0]
                        car_model_name = vehicle_system_model_list[1]
                    if vehicle_system_name in self.system_name_dict:
                        vehicle_system_name = self.system_name_dict[vehicle_system_name]
                    # 奥迪RS 5 , Sport 奥迪RS 5  这种情况，车系匹配不上
                    # 通过品牌id获取对应的车系名称及id
                    self.cursor_default_tow.execute(self.query_car_system_sql.format(brand_id))
                    vehicle_system_list = self.cursor_default_tow.fetchall()
                    for vehicle_system in vehicle_system_list:
                        vehicle_system_name_new = vehicle_system.get("name")
                        if brand_name == "奥迪":
                            vehicle_system_name_new = str(vehicle_system_name_new).replace('Sport', '').strip()
                            vehicle_system_name = str(vehicle_system_name).replace('Sport', '').strip()
                        if str(vehicle_system_name).upper().replace(brand_name, '') == str(
                                vehicle_system_name_new).upper().replace(brand_name, ''):
                            # 确保车系名字和业务表中存的是一致的
                            vehicle_system_name = vehicle_system.get("name")
                            vehicle_system_id = vehicle_system.get("id")
                            break
                    if not vehicle_system_id:
                        self.update_mongodb(2, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue

                elif spider_name == "second_hand_car_guazi":
                    # [北京奇瑞二手车] 通过结巴分词区分出车品牌
                    brand_name = ''.join(content.get("car_brand")).replace('二手车', '')
                    brand_name = jieba.cut(brand_name)
                    brand_name_list = ','.join(brand_name).split(',')
                    # print(brand_name_list)
                    # 唐山宝骏用结巴分词分不开
                    # 唐山哈弗也分不开,所以对唐山车进行特殊处理
                    if "唐山" in ''.join(content.get("car_brand")).replace('二手车', ''):
                        brand_name = ''.join(content.get("car_brand")).replace('二手车', '').replace('唐山', '')
                    # ['杭州路', '虎'] ['南京路', '虎'] ['北京路', '虎']
                    elif "路虎" in ''.join(content.get("car_brand")):
                        brand_name = "路虎"
                    # ['上', '海陆风']
                    elif "陆风" in ''.join(content.get("car_brand")):
                        brand_name = "陆风"
                    # ['上海大众']
                    elif len(brand_name_list) == 1 and "上海大众" == brand_name_list[0]:
                        brand_name = "大众"
                    elif len(brand_name_list) == 2:
                        brand_name = brand_name_list[1]
                    # ['安庆', '东风', '小康']
                    elif len(brand_name_list) == 3:
                        brand_name = brand_name_list[1] + brand_name_list[2]
                    # ['安庆', '江铃', '集团', '新能源']
                    elif len(brand_name_list) == 4:
                        brand_name = brand_name_list[1] + brand_name_list[2] + brand_name_list[3]

                    # 将车品牌统一成业务表中存的数据
                    for brand_name_k, brand_name_v in self.brand_name_dict.items():
                        if brand_name == brand_name_k:
                            brand_name = brand_name_v
                            break

                    self.cursor_default_tow.execute(self.query_car_brand_id_sql.format(brand_name, 0))
                    brand_id_list = self.cursor_default_tow.fetchall()
                    # 判断车型表中是否存在该品牌，存在获取id
                    if brand_id_list:
                        brand_id = brand_id_list[0].get("id")
                    else:
                        self.update_mongodb(2, mog_id, collection)
                        # 当于业务表中的车品牌对应不上的时候，跳过，并把remake状态设置为2
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue
                    """
                    奇瑞 风云2 2013款 两厢 1.5L 手动锐意版
                    宝马3系 2013款 改款 328Li 豪华设计套装
                    哈弗H6 2012款 1.5T 手动两驱精英型  title可能包括车品牌和车系，也可能只包括车系
                    雪佛兰 2015款 赛欧3 1.5L 手动理想天窗版 
                    哈弗H6 Coupe 2016款 蓝标 1.5T 自动两驱精英型
                    获取车型和车系 将两个空格换成一个空格
                    # Smart smart fortwo 2012款 1.0 MHD 硬顶舒适版 需要单独处理
                    """
                    title = content.get("title").replace("  ", " ")
                    # 根据不同情况过滤title
                    title = self.filter_title_rule(title, brand_name, source)

                    # 按空格切割title字段
                    vehicle_system_model_list = title.split(' ', 2)
                    if brand_name == "雪佛兰" and "赛欧3" in title:
                        vehicle_system_name = "赛欧"
                        car_model_name = vehicle_system_model_list[1] + " " + vehicle_system_model_list[2]
                    # 丰田 YARiS L 致炫 2016款 改款 1.5E CVT魅动版
                    elif brand_name == "丰田" and "YARiS L 致炫" in title:
                        vehicle_system_name = "YARiS L 致炫"
                        car_model_name = title.split(" ", 4)[4]

                    # 大众 2016款 途安L 280TSI 自动舒雅版 获取不到车系，需要单独处理
                    elif brand_name == "大众" and "途安" in title:
                        vehicle_system_name = "途安"
                        car_model_name = vehicle_system_model_list[1] + " " + vehicle_system_model_list[2]
                    elif len(vehicle_system_model_list) == 3 and '款' in str(vehicle_system_model_list[2]).replace('改款',
                                                                                                                  '') \
                            or len(vehicle_system_model_list) == 3 and '款' not in vehicle_system_model_list[
                        1] and '款' not in \
                            str(vehicle_system_model_list[2]).replace('改款', ''):
                        vehicle_system_name = vehicle_system_model_list[1]
                        # 解决【众泰T600 Coupe】【哈弗H6 Coupe】为车系名称的情况
                        vehicle_system_name_other = vehicle_system_model_list[0] + " " + vehicle_system_model_list[1]
                        car_model_name = vehicle_system_model_list[2]
                    elif len(vehicle_system_model_list) == 3 and '款' in vehicle_system_model_list[1]:
                        vehicle_system_name = vehicle_system_model_list[0]
                        car_model_name = vehicle_system_model_list[1] + ' ' + vehicle_system_model_list[2]
                    elif len(vehicle_system_model_list) == 2:
                        vehicle_system_name = vehicle_system_model_list[0]
                        car_model_name = vehicle_system_model_list[1]
                    if vehicle_system_name in self.system_name_dict:
                        vehicle_system_name = self.system_name_dict[vehicle_system_name]
                    # 通过品牌id获取对应的车系名称及id
                    self.cursor_default_tow.execute(self.query_car_system_sql.format(brand_id))
                    vehicle_system_list = self.cursor_default_tow.fetchall()
                    for vehicle_system in vehicle_system_list:
                        vehicle_system_name_new = vehicle_system.get("name")
                        # Sport 奥迪R8 和 奥迪R8 同一个车系
                        # Sport 奥迪RS 3
                        if brand_name == "奥迪":
                            vehicle_system_name_new = str(vehicle_system_name_new).replace('Sport', '').replace(' ', '')
                            vehicle_system_name = str(vehicle_system_name).replace('Sport', '').replace(' ', '')
                        # 奔驰GLA 和 奔驰GLA级
                        elif brand_name == "奔驰":
                            vehicle_system_name_new = str(vehicle_system_name_new).replace("级", '')
                            vehicle_system_name = str(vehicle_system_name).replace("级", '')
                        # 中华骏捷 和 骏捷
                        elif brand_name == "中华":
                            vehicle_system_name_new = str(vehicle_system_name_new).replace("中华", '')
                            vehicle_system_name = str(vehicle_system_name).replace("中华", '')
                        # 北京现代ix25 和 现代ix25
                        elif brand_name == "现代":
                            vehicle_system_name_new = str(vehicle_system_name_new).replace("北京", '')
                            vehicle_system_name = str(vehicle_system_name).replace("北京", '')

                        # 将车系名称统一 【奔驰唯雅诺 和 唯雅诺】
                        if str(vehicle_system_name).upper().replace(brand_name, '') == \
                                str(vehicle_system_name_new).upper().replace(brand_name, '') or \
                                str(vehicle_system_name_new).upper().replace(brand_name, '') == \
                                str(vehicle_system_name_other).upper().replace(brand_name, ''):
                            # 确保车系名字和业务表中存的是一致的
                            vehicle_system_name = vehicle_system.get("name")
                            vehicle_system_id = vehicle_system.get("id")
                            break
                    if not vehicle_system_id:
                        self.update_mongodb(2, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue

                elif spider_name == "second_hand_car_taoche":
                    if content.get("car_brand"):
                        brand_name = content.get("car_brand")
                        if brand_name == "传祺":
                            brand_name = "广汽传祺"
                    self.cursor_default_tow.execute(self.query_car_brand_id_sql.format(brand_name, 0))
                    brand_id_list = self.cursor_default_tow.fetchall()
                    # 判断车型表中是否存在该品牌，存在获取id
                    if brand_id_list:
                        brand_id = brand_id_list[0].get("id")
                    else:
                        self.update_mongodb(2, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue
                    # 安庆 宜秀区-V3菱悦 2012款 改款 1.5L 手动 豪华版
                    title = str(content.get("title")).split('-')[1]
                    vehicle_system_model_list = title.split(" ", 2)
                    if len(vehicle_system_model_list) == 3:
                        if '款' in str(vehicle_system_model_list[1]):
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                        elif '款' not in str(vehicle_system_model_list[1]) and '款' in str(vehicle_system_model_list[2]):
                            vehicle_system_name = str(vehicle_system_model_list[0]) + " " + str(
                                vehicle_system_model_list[1])
                            car_model_name = vehicle_system_model_list[2]
                        else:
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                    elif len(vehicle_system_model_list) == 2:
                        vehicle_system_name = vehicle_system_model_list[0]
                        car_model_name = vehicle_system_model_list[1]
                    if vehicle_system_name in self.system_name_dict:
                        vehicle_system_name = self.system_name_dict[vehicle_system_name]
                    self.cursor_default_tow.execute(self.query_car_system_sql.format(brand_id))
                    vehicle_system_list = self.cursor_default_tow.fetchall()
                    for vehicle_system in vehicle_system_list:
                        if str(vehicle_system_name).upper().replace(brand_name, '') == str(
                                vehicle_system.get("name")).upper().replace(brand_name, ''):
                            # 确保车系名字和业务表中存的是一致的
                            vehicle_system_name = vehicle_system.get("name")
                            vehicle_system_id = vehicle_system.get("id")
                            break
                    if not vehicle_system_id:
                        self.update_mongodb(2, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '2', used_car['id']))
                        # self.conn_default.commit()
                        continue

                elif spider_name == "second_hand_car_xin":
                    # 生产厂商
                    manufacturer = content.get("car_brand")
                    # 东风风光 风光330 2017款 1.5 手动 实用型II DK15-02
                    # 奔驰 B级 2015款 1.6T 自动 B200动感型  2015款 B 200 动感型
                    title = str(content.get("title")).split(' ', 3)
                    brand_name = title[0]
                    # 统一车品牌名称
                    if brand_name in self.brand_name_dict:
                        brand_name = self.brand_name_dict[brand_name]

                    if len(title) == 4:
                        if '款' in str(title[2]):
                            vehicle_system_name = title[1]
                            car_model_name = str(title[2]) + ' ' + str(title[3])
                        elif '款' not in str(title[2]) and '款' in str(title[3]):
                            vehicle_system_name = str(title[1]) + " " + str(title[2])
                            car_model_name = title[3]
                        else:
                            vehicle_system_name = title[1]
                            car_model_name = str(title[2]) + ' ' + str(title[3])
                    elif len(title) == 3:
                        vehicle_system_name = title[1]
                        car_model_name = title[2]
                    if vehicle_system_name in self.system_name_dict:
                        vehicle_system_name = self.system_name_dict[vehicle_system_name]
                    vehicle_system_name_new = str(brand_name) + str(vehicle_system_name)
                    # 统一车系名称
                    if "进口" in manufacturer:
                        # 存在宝马车vehicle_system_name是1系，vehicle_system_name_new是1系进口都能对应上的情况，优先取进口名称车系
                        vehicle_system_name = self.manage_car_system_youxin(vehicle_system_name)
                        vehicle_system_name_new = self.manage_car_system_youxin(vehicle_system_name_new)
                        if "进口" in vehicle_system_name:
                            vehicle_system_name_new = vehicle_system_name
                        elif "进口" in vehicle_system_name_new:
                            vehicle_system_name = vehicle_system_name_new
                    # 处理在车系中有"厢"字的情况
                    if "厢" in str(vehicle_system_name):
                        vehicle_system_name = self.manage_car_system_youxin_compartments(vehicle_system_name)
                        vehicle_system_name_new = self.manage_car_system_youxin_compartments(vehicle_system_name_new)

                    self.cursor_default_tow.execute(self.query_car_brand_id_sql.format(brand_name, 0))
                    brand_id_list = self.cursor_default_tow.fetchall()
                    # 判断车型表中是否存在该品牌，存在获取id
                    if brand_id_list:
                        brand_id = brand_id_list[0].get("id")
                    else:
                        self.update_mongodb(4, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '4', used_car['id']))
                        # self.conn_default.commit()
                        continue

                    self.cursor_default_tow.execute(self.query_car_system_sql.format(brand_id))
                    vehicle_system_list = self.cursor_default_tow.fetchall()
                    for vehicle_system in vehicle_system_list:
                        if str(vehicle_system_name).upper().replace(brand_name, '') == str(vehicle_system.get(
                                "name")).upper().replace(brand_name, '') or str(vehicle_system_name_new).upper(). \
                                replace(brand_name, '') == str(vehicle_system.get("name")).upper().replace(brand_name,
                                                                                                           ''):
                            # 确保车系名字和业务表中存的是一致的
                            vehicle_system_name = vehicle_system.get("name")
                            vehicle_system_id = vehicle_system.get("id")
                            break
                    if not vehicle_system_id:
                        self.update_mongodb(5, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '5', used_car['id']))
                        # self.conn_default.commit()
                        continue

                elif spider_name == "second_hand_car_renrenche":
                    """
                    title中不同的情况:
                        宝马X1(进口) 2012款 sDrive18i豪华型
                        新凯汽车-凌特 2015款 3.5L 标准型
                        凯迪拉克ATS-L 2016款 28T 技术型
                        本田-CR-V 2004款 2.0L
                        福特-蒙迪欧-致胜 2011款 2.3L 豪华型
                    """
                    title = content.get("title")
                    title_list = str(title).split('-')
                    # 奥迪-A4 2006款 2.0T 尊享型
                    # 获取车品牌
                    brand_name = title_list[0]
                    # 处理车品牌名称
                    brand_name = self.manage_car_brand(brand_name, title)
                    # 区分进口车
                    title = self.filter_title_rule(title, brand_name, source).split('-')
                    # 将车品牌统一成业务表中存的数据
                    if self.brand_name_dict.get(brand_name):
                        brand_name = self.brand_name_dict.get(brand_name)

                    self.cursor_default_tow.execute(self.query_car_brand_id_sql.format(brand_name, 0))
                    brand_id_list = self.cursor_default_tow.fetchall()
                    # 判断车型表中是否存在该品牌，存在获取id
                    if brand_id_list:
                        brand_id = brand_id_list[0].get("id")
                    else:
                        self.update_mongodb(4, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '4', used_car['id']))
                        # self.conn_default.commit()
                        continue

                    # 获取车系
                    if len(title) == 1:
                        title_new = title[0]
                    else:
                        title_new = "".join(title[1:])

                    vehicle_system_model_list = title_new.split(" ", 2)
                    if len(vehicle_system_model_list) == 3:
                        if '款' in str(vehicle_system_model_list[1]):
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                        elif '款' not in str(vehicle_system_model_list[1]) and '款' in str(vehicle_system_model_list[2]):
                            vehicle_system_name = str(vehicle_system_model_list[0]) + " " + str(
                                vehicle_system_model_list[1])
                            vehicle_system_name_one = vehicle_system_model_list[1]
                            car_model_name = vehicle_system_model_list[2]
                        else:
                            vehicle_system_name = vehicle_system_model_list[0]
                            car_model_name = str(vehicle_system_model_list[1]) + ' ' + str(vehicle_system_model_list[2])
                    elif len(vehicle_system_model_list) == 2:
                        vehicle_system_name = vehicle_system_model_list[0]
                        car_model_name = vehicle_system_model_list[1]
                    if vehicle_system_name in self.system_name_dict:
                        vehicle_system_name = self.system_name_dict[vehicle_system_name]
                    self.cursor_default_tow.execute(self.query_car_system_sql.format(brand_id))
                    vehicle_system_list = self.cursor_default_tow.fetchall()
                    # 奥迪-A4 这种情况需要拼接
                    vehicle_system_name_new = str(title_list[0]) + str(vehicle_system_name)
                    vehicle_system_name_new = self.manage_car_system(brand_name, vehicle_system_name_new)
                    for vehicle_system in vehicle_system_list:
                        vehicle_system_name_ = vehicle_system.get("name")
                        # 处理车系名称
                        vehicle_system_name_ = self.manage_car_system(brand_name, vehicle_system_name_)
                        if vehicle_system_name:
                            vehicle_system_name = self.manage_car_system(brand_name, vehicle_system_name)
                        if vehicle_system_name_one:
                            vehicle_system_name_one = self.manage_car_system(brand_name, vehicle_system_name_one)
                        if str(vehicle_system_name) == str(vehicle_system_name_).replace('-', '') \
                                or str(vehicle_system_name_new) == str(vehicle_system_name_).replace('-', '') \
                                or str(vehicle_system_name_one) == str(vehicle_system_name_).replace('-', ''):
                            # 确保车系名字和业务表中存的是一致的
                            vehicle_system_name = vehicle_system.get("name")
                            vehicle_system_id = vehicle_system.get("id")
                            break
                    if not vehicle_system_id:
                        self.update_mongodb(5, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '5', used_car['id']))
                        # self.conn_default.commit()
                        continue

                # 通过车系id获取对应的车型名称及id
                self.cursor_default_tow.execute(self.query_car_system_sql.format(vehicle_system_id))
                car_model_list = self.cursor_default_tow.fetchall()
                car_model_data = self.get_car_model_data(car_model_list, car_model_name, model_year)
                # 获取匹配上的车型和车型id，如果匹配不上会返回空值
                car_model_name = car_model_data.get("car_model_name")
                car_model_id = car_model_data.get("car_model_id")
                # 如果没有匹配上对应的车型id，断开这次循环，继续下一次循环且改变remark为2
                if not car_model_name or not car_model_id:
                    self.update_mongodb(6, mog_id, collection)
                    # self.cursor_default.execute(self.update_remark_sql.format(table_name, '6', used_car['id']))
                    # self.conn_default.commit()
                    continue

                # 车类别
                car_class, new_level = self.manage_car_class(level, car_model_id)
                # 去重操作
                # try:
                #     self.cursor_default_tow.execute(self.repeat_sql.format(
                #         price=price, meter_mile=meter_mile, semiautomatic_gearbox=semiautomatic_gearbox,
                #         displacement=displacement, emission_standard=emission_standard, sell_times=sell_times,
                #         year_check_end_time=year_check_end_time, is_have_strong_risk=is_have_strong_risk,
                #         type=level, brand_id=brand_id, vehicle_system_id=vehicle_system_id,
                #         car_model_id=car_model_id))
                # except Exception:
                #     logging.error("错误日志：{0},id:{1}".format(traceback.format_exc(), mog_id))
                #     self.update_mongodb(7, mog_id, collection)
                # self.cursor_default.execute(self.update_remark_sql.format(table_name, '7', used_car['id']))
                # self.conn_default.commit()
                # continue
                count_num = self.duplicate_removal(price, meter_mile, semiautomatic_gearbox, displacement,
                                                   emission_standard, sell_times, year_check_end_time,
                                                   is_have_strong_risk, level, brand_id, vehicle_system_id, car_model_id)

                # count_num = self.cursor_default_tow.fetchall()[0]['count']
                if not count_num:
                    try:
                        self.cursor_default_tow.execute(self.insert_car_sell_sql, (
                            remark, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), None, None,
                            isDeleted, source, price, register_time, meter_mile, semiautomatic_gearbox, displacement,
                            emission_standard, sell_times, year_check_end_time, is_have_strong_risk,
                            business_risk_end_time, img_url, risk_27_check, level, detail_url, brand_id,
                            vehicle_system_id,car_model_id, brand_name, vehicle_system_name, car_model_name,
                            car_class, model_year, new_level))
                        self.default_tow.commit()
                    except Exception:
                        logging.error("错误日志：{0},id:{1}".format(traceback.format_exc(), mog_id))
                        self.update_mongodb(7, mog_id, collection)
                        # self.cursor_default.execute(self.update_remark_sql.format(table_name, '7', used_car['id']))
                        # self.conn_default.commit()
                        continue
                self.update_mongodb(1, mog_id, collection)
                # self.cursor_default.execute(self.update_remark_sql.format(table_name, '1', used_car['id']))
                # self.conn_default.commit()
            # 删除索引
            collection.drop_index("id_1")

    # 处理汽车分类数据
    def manage_car_class(self, level, model_id):
        car_class = None
        new_level = None
        # 先从新车表中查看对应的model_id，如果存在获取对应的car_class
        self.cursor_default_tow.execute(self.get_car_class.format(model_id))
        car_class_list = self.cursor_default_tow.fetchall()
        if car_class_list:
            car_class = car_class_list[0]['car_class']
            new_level = car_class_list[0]['level']
        else:
            # 将车类型划分成多个类别
            for k, v in self.car_class_dict.items():
                if str(level) in v:
                    car_class = k
                    break
        return car_class, new_level

    # 处理车品牌数据
    def manage_car_brand(self, brand_name: str, title: str):
        if "凯迪拉克" in title:
            brand_name = "凯迪拉克"
        elif "宝马" in title:
            brand_name = "宝马"
        elif brand_name == "北汽" and "幻速" in title:
            brand_name = "北汽幻速"
        elif brand_name == "北汽" and "绅宝" in title:
            brand_name = "北汽绅宝"
        elif brand_name == "北汽" and "威旺" in title:
            brand_name = "北汽威旺"
        elif brand_name == "广汽" and "传祺" in title:
            brand_name = "广汽传祺"
        elif brand_name == "东风" and "风光" in title:
            brand_name = "东风风光"
        elif brand_name == "东风" and "小康" in title:
            brand_name = "东风小康"
        elif brand_name == "东风" and "风神" in title:
            brand_name = "东风风神"
        elif "英致" in title:
            brand_name = "潍柴英致"
        return brand_name

    # 处理车系数据
    def manage_car_system(self, brand_name: str, vehicle_system_name):
        # Sport 奥迪R8 和 奥迪R8 同一个车系
        # Sport 奥迪RS 3
        if brand_name == "奥迪":
            vehicle_system_name = str(vehicle_system_name).replace('Sport', '').replace(' ', '')
        # 奔驰GLA 和 奔驰GLA级
        elif brand_name == "奔驰":
            vehicle_system_name = str(vehicle_system_name).replace("级", '')
        # 中华骏捷 和 骏捷
        elif brand_name == "中华":
            vehicle_system_name = str(vehicle_system_name).replace("中华", '')
        # 北京现代ix25 和 现代ix25
        elif brand_name == "现代":
            vehicle_system_name = str(vehicle_system_name).replace("北京", '')
        # 广汽传祺-GS5 和 传祺GS5
        elif brand_name == "广汽传祺":
            vehicle_system_name = str(vehicle_system_name).replace("广汽", '')
        elif brand_name == "北汽绅宝":
            vehicle_system_name = str(vehicle_system_name).replace("北汽", '')

        return vehicle_system_name.replace("汽车", "").replace(brand_name.replace("汽车", ""), '').upper()

    # 处理优信网车系数据
    def manage_car_system_youxin(self, car_system):
        car_system_dict = {"劲炫": "ASX劲炫(进口)", "欧蓝德": "欧蓝德(进口)", "帕杰罗": "帕杰罗(进口)",
                           "帕杰罗·劲畅": "帕杰罗·劲畅(进口)", "太空车": "太空车（进口）", "吉姆尼": "吉姆尼(进口)",
                           "凯迪拉克ATS": "凯迪拉克ATS(进口)", "凯迪拉克CTS": "凯迪拉克CTS(进口)", "RAV4": "RAV4（进口）",
                           "普拉多": "普拉多(进口)", "汉兰达": "汉兰达(进口)", "兰德酷路泽": "兰德酷路泽(进口)",
                           "凯美瑞": "凯美瑞(海外)", "高尔夫": "高尔夫(进口)", "迈腾": "迈腾(进口)", "R36": "R36（进口）",
                           "高尔夫新能源": "高尔夫新能源(进口)", "宝马5系": "宝马5系(进口)", "宝马1系": "宝马1系(进口)",
                           "宝马3系": "宝马3系(进口)", "宝马X3": "宝马3系(进口)", "宝马X1": "宝马X1(进口)",
                           "宝马2系旅行车": "宝马2系旅行车(进口)", "威霆": "威霆(进口)", "奔驰C级": "奔驰C级(进口)",
                           "奔驰A级": "奔驰A级(进口)", "唯雅诺": "唯雅诺(进口)", "奔驰E级": "奔驰E级(进口)",
                           "奔驰GLA": "奔驰GLA(进口)", "奔驰GLC": "奔驰GLC(进口)", "奥迪A3": "奥迪A3(进口)",
                           "奥迪A4": "奥迪A4(进口)", "奥迪A6": "奥迪A6(进口)", "奥迪Q5": "奥迪Q5(进口)",
                           "奥迪A3新能源": "奥迪A3新能源(进口)", "奥迪Q3": "奥迪Q3(进口)", "R8": "R8（进口）",
                           "揽胜极光": "揽胜极光(进口)", "发现神行": "发现神行(进口)", "指南者": "指南者(进口)",
                           "自由光": "自由光(进口)", "大切诺基": "大切诺基(进口)", "飞度": "飞度(进口)",
                           "标致206": "标致206(进口)", "标致307": "标致307(进口)", "标致207": "标致207(进口)",
                           "标致308": "标致308(进口)", "标致3008": "标致3008(进口)", "标致4008": "标致4008(进口)",
                           "DS 5": "DS 5（进口）", "锐界": "锐界(进口)", "福克斯": "福克斯(进口)", "嘉年华": "嘉年华(进口)",
                           "克莱斯勒300C": "克莱斯勒300C(进口)", "大捷龙": "大捷龙(进口)", "大捷龙PHEV": "大捷龙PHEV(进口)",
                           "克莱斯勒君王": "克莱斯勒君王(进口)", "科雷傲": "科雷傲(进口)", "马自达8": "马自达8(进口)",
                           "马自达3": "马自达3(进口)", "马自达CX-7": "马自达CX-7(进口)", "马自达CX-5": "马自达CX-5(进口)",
                           "讴歌RDX": "讴歌RDX(进口)", "嘉华": "嘉华(进口)", "起亚K5": "起亚K5新能源", "奇骏": "奇骏(进口)",
                           "楼兰": "楼兰(海外)", "昊锐": "昊锐(进口)", "Yeti": "Yeti(进口)", "明锐": "明锐(进口)",
                           "速派": "速派(进口)", "沃尔沃S40": "沃尔沃S40(进口)", "沃尔沃XC60": "沃尔沃XC60(进口)",
                           "沃尔沃S90": "沃尔沃S90(进口)", "途胜": "途胜(进口)", "胜达": "胜达(进口)", "索纳塔": "索纳塔(进口)",
                           "科帕奇": "科帕奇(进口)", "雪铁龙C4 Aircross": "雪铁龙C4 Aircross(进口)", "雪铁龙C5": "雪铁龙C5(进口)",
                           "雪铁龙C6": "雪铁龙C6(进口)", "英菲尼迪QX50": "英菲尼迪QX50(进口)"}

        if car_system in car_system_dict:
            car_system = car_system_dict[car_system]
        return car_system

    def manage_car_system_youxin_compartments(self, car_system):
        if "两厢" in car_system:
            car_system = car_system.split("两厢")[0]
        if "三厢" in car_system:
            car_system = car_system.split("三厢")[0]
        return car_system

    # 过滤年检到期时间
    def filter_year_check_end_time(self, year_check_end_time: str):
        flag = 0
        now_year = datetime.datetime.now().year
        year_future = now_year + 6
        year_old = now_year - 20
        year_check = str(year_check_end_time).split("-")[0]
        if int(year_check) > year_future or int(year_check) < year_old:
            flag = 1
        return flag

    # 处理上牌时间数据
    def manage_register_time(self, register_time):
        re_register = re.compile(r"\d+")
        time_list = re_register.findall(register_time)
        # ['2015', '11']
        if time_list and len(time_list) == 2:
            if len(time_list[0]) == 4:
                register_time = "-".join(time_list) + "-01"
        return register_time

    # 修改remark状态
    def update_mongodb(self, num: int, mog_id: int, collection):
        """
        :param num: 根据传入的数字不同，修改mongodb不同的状态
        :param mog_id: 修改的数据id
        :param collection:
        :return: None
        """
        collection.update_one({'id': mog_id}, {"$set": {"is_Processed": num}})

    # 根据不同情况区分去重条件
    def duplicate_removal(self, price, meter_mile, semiautomatic_gearbox, displacement, emission_standard, sell_times,
                          year_check_end_time, is_have_strong_risk, level, brand_id, vehicle_system_id, car_model_id):
        count_num = None
        # 1.都存在
        one_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s 
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times = %s
                and year_check_end_time = %s
                and is_have_strong_risk = %s
                and type = %s and brand_id = %s and vehicle_system_id = %s 
                and car_model_id = %s"""
        # 2 都不存在
        two_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s 
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times is NULL
                and year_check_end_time is NULL
                and is_have_strong_risk is NULL 
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 3.过户次数存在，年检到期时间和交强险时间不存在
        three_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times = %s 
                and year_check_end_time is NULL
                and is_have_strong_risk is NULL
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 4.年检到期时间存在，过户次数和交强险时间不存在
        four_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times is NULL
                and year_check_end_time = %s
                and is_have_strong_risk is NULL 
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 5.交强险时间存在，过户次数和年检到期时间不存在
        five_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times is NULL 
                and year_check_end_time is NULL 
                and is_have_strong_risk = %s
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 6.过户次数和交强险时间存在，年检到期时间不存在
        six_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times = %s
                and year_check_end_time is NULL
                and is_have_strong_risk = %s
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 7.过户次数和年检到期时间存在，交强险时间不存在
        seven_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s
                and sell_times = %s
                and year_check_end_time = %s
                and is_have_strong_risk is NULL
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        # 8.交强险时间和年检到期时间存在，过户次数不存在
        eight_situation = """select IFNULL(count(*),0) as count,id from second_car_sell where price = %s
                and meter_mile = %s and semiautomatic_gearbox = %s
                and displacement = %s and emission_standard = %s 
                and sell_times is NULL 
                and year_check_end_time = %s
                and is_have_strong_risk = %s
                and type = %s and brand_id = %s and vehicle_system_id = %s
                and car_model_id = %s"""
        if sell_times and year_check_end_time and is_have_strong_risk:
            self.cursor_default_tow.execute(one_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard, sell_times,
                year_check_end_time, is_have_strong_risk,
                level, brand_id, vehicle_system_id,
                car_model_id))

        elif not sell_times and not year_check_end_time and not is_have_strong_risk:
            self.cursor_default_tow.execute(two_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif sell_times and not year_check_end_time and not is_have_strong_risk:
            self.cursor_default_tow.execute(three_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard, sell_times,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif year_check_end_time and not sell_times and not is_have_strong_risk:
            self.cursor_default_tow.execute(four_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard,
                year_check_end_time,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif is_have_strong_risk and not sell_times and not year_check_end_time:
            self.cursor_default_tow.execute(five_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard,
                is_have_strong_risk,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif sell_times and is_have_strong_risk and not year_check_end_time:
            self.cursor_default_tow.execute(six_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard, sell_times,
                is_have_strong_risk,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif sell_times and year_check_end_time and not is_have_strong_risk:
            self.cursor_default_tow.execute(seven_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard, sell_times,
                year_check_end_time,
                level, brand_id, vehicle_system_id,
                car_model_id))
        elif is_have_strong_risk and year_check_end_time and not sell_times:
            self.cursor_default_tow.execute(eight_situation, (
                price, meter_mile, semiautomatic_gearbox,
                displacement, emission_standard,
                year_check_end_time, is_have_strong_risk,
                level, brand_id, vehicle_system_id,
                car_model_id))
        count_num = self.cursor_default_tow.fetchall()[0]['count']
        return count_num


# --table-name base_data_second_hand_car --db-name valuation_web
if __name__ == '__main__':
    if not db_name:
        logging.error("--db-name参数不能为空，例如：valuation_web")
        sys.exit(1)
    elif not table_name:
        logging.error("--table-name参数不能为空，例如：base_data_second_hand_car")
        sys.exit(1)

    cleanDataIntoCar = CleanDataIntoCar()
    cleanDataIntoCar.clean_car_sell_data()

    cleanDataIntoCar.cursor_default.close()
    cleanDataIntoCar.conn_default.close()
    cleanDataIntoCar.cursor_default_tow.close()
    cleanDataIntoCar.default_tow.close()

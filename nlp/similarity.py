#!/usr/bin/python3
# -*- coding: utf-8 -*-


import pymysql
import re
from pprint import pprint
import jieba
import argparse
from gensim import corpora, models, similarities


parser = argparse.ArgumentParser(description='命令行参数测试')
parser.add_argument('--stopwords-path', type=str, default='')
parser.add_argument('--userdict-path', type=str, default='')
args = parser.parse_args()
stopwords_path = args.stopwords_path
userdict_path = args.userdict_path


class Similarity():
    '''
    相似度匹配
    '''

    def __init__(self):
        self.default_conn = pymysql.connect(host='***', user='***', passwd='***',
                                            db='***', port=3306, charset='utf8',
                                            cursorclass=pymysql.cursors.DictCursor, connect_timeout=7200)
        self.default_cursor = self.default_conn.cursor()

    def get_text(self, file_name):
        """
        获取文本数据
        :param file_name: 文本路径
        :return: 文本内容
        """
        f = open(file_name, 'r', encoding='utf-8')
        text = f.read()
        f.close()
        return text

    def gen_corpus(self, documents):
        """
        生成document
        :param documents: 原生
        :return: docs
        """
        stop_words = self.get_text(stopwords_path)
        docs = re.sub('\\（.*?\\）+|\\(.*?\\)', '', documents).replace(' ', '')
        docs = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:：。？、~@#￥%……&*（）]+", ",", docs)
        doc_list = [doc for doc in docs.split(',') if doc not in stop_words]
        return doc_list

    def segment(self, sentence):
        """
        分词
        :param sentence: 待分内容
        :return: 词列表
        """
        jieba.load_userdict(userdict_path)
        sentence = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", sentence)
        # stopwords = self.get_text(stopwords_path)
        segResult = jieba.lcut(sentence, HMM=True)
        words = []
        for word in segResult:
            # if word in stopwords:
            #     continue
            if word.isspace():
                continue
            else:
                words.append(word)
        return words

    def sim_lsi(self, target_doc, docs):
        """
        文本相似度计算
        :param target_doc: 目标文本
        :param docs: 语料文本
        :return:
        """
        # 语料文本分词生成语料库
        texts = [self.segment(doc) for doc in docs]
        # pprint(texts)
        # 基于文件集建立词典，并提取词典特征数
        dictionary = corpora.Dictionary(texts)

        # 目标文本分词
        target_words = self.segment(target_doc)
        # pprint(target_words)
        target_vec_bow = dictionary.doc2bow(target_words)

        # 基于词典，将【分词列表集】转换为【稀疏向量集】
        corpus = [dictionary.doc2bow(text) for text in texts]
        # lsi = models.LsiModel(corpus, id2word=dictionary)

        tfidf = models.TfidfModel(corpus)

        # lsi[corpus]==>Lsi训练好的语料库模型， index是设定的匹配相识度的条件
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
        # 矩阵相似度计算
        sim = index[tfidf[target_vec_bow]]
        sim = sorted(enumerate(sim), key=lambda item: -item[1])
        for s in sim:
            print(s, docs[s[0]])

    def run(self):
        target = input('\n请输入待匹配文本： ') or '四、汇率变动对                      现金及现金等价物的影响'
        sheet_type = input('\n其属于： \n') or '现金流量表'
        sql_to_phrase = """select special_match, total_row from report_match_relation where sheet_type='{0}'"""
        self.default_cursor.execute(sql_to_phrase.format(sheet_type))
        documents = self.default_cursor.fetchall()[0]
        pprint(documents)
        docs = documents['special_match'] + ',' + documents['total_row']
        docs = list({}.fromkeys(self.gen_corpus(docs)).keys())
        pprint(docs)

        self.sim_lsi(target, docs)


if __name__ == '__main__':
    sim = Similarity()
    phrase = sim.run()

#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
import docx
from pyltp import SentenceSplitter
from pyltp import Segmentor
from snownlp import normal
from snownlp.summary import textrank
import xmnlp


def get_text(file_name):
    '''
    Desc: 读取文本内容
    Args: file_name 文本文件位置
    Ｒeturns: text(str) 文本内容
    '''

    f = open(file_name, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    return text

def get_doc(file_name):
    '''
    Desc: 读取Word文档
    Args: file_name 文档路径
    Returns: document(str) 文档字符串
    '''
    
    file = docx.Document(file_name)
    graphs = [graph.text.replace(' ', '') for graph in file.paragraphs]
    document = '\n'.join(graphs)
    return document

def doc2sent(doc):
    '''
    Desc: 除去文档中的空行并组成字符串
    Args: doc 文档字符串
    Returns: sents(list) 句子列表
    '''
    
    sentences = SentenceSplitter.split(doc)  # 切分成句子
    sents = [sent for sent in sentences if sent != '']
    return sents

def segment_jieba(sentence, stopwords):
    '''
    Desc: 分词并除去停用词
    Args: sentence 待分词的文本内容
          stopwords 停用词
    Returns: words 分词结果
    '''

    jieba.load_userdict('/home/kdd/nlp/userdict.txt')
    # sentence = ' '.join(sentence)
    segResult = jieba.lcut(sentence)
    words = []
    for word in segResult:
        if word in stopwords:
            continue
        elif word.isspace():
            continue
        else:
            words.append(word)
    return words

def segment_ltp(text, stopwords_file, cws_model_path):
    '''
    Desc: LTP分词
    Args: text　待分词文本
          stopwords_file 停用词
          cws_model_path 分词模型
    Returns: words 分词列表
    '''
    segmentor = Segmentor()
    segmentor.load_with_lexicon(cws_model_path, stopwords_file) # 加载模型、自定义词典
    words = segmentor.segment(text)
    segmentor.release()
    return words


# 自定义文本位置
text_file = '/home/kdd/nlp/业务约定书.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'



if __name__ == '__main__':

    # 获取文本和停用词
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    # print(doc)
    
    # 切分成句子
    sents = doc2sent(doc)  
    # print(sents)

    # 摘要提取
    word_list = [segment_jieba(sent, stopWords) for sent in sents]
    # print(word_list)

    rank = textrank.TextRank(word_list)
    rank.solve()
    # key_sents = [sents[index] for index in rank.top_index(5)]
    # print(key_sents)
    for index in rank.top_index(5):
        print(sents[index])
    
    # keyword_rank = textrank.KeywordTextRank(word_list)
    # keyword_rank.solve()
    # for w in keyword_rank.top_index(10):
    #     print(w)

    # xmnlp摘要提取
    xmnlp.set_stopword('/home/kdd/nlp/stop_words.txt')
    xmnlp.set_userdict('/home/kdd/nlp/userdict.txt')
    t = xmnlp.keyphrase(doc)
    print(t)
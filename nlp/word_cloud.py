#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
from jieba import posseg as psg
from jieba import analyse
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import docx
from pyltp import SentenceSplitter
from pyltp import Segmentor


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

    doc = re.sub('\\(.*?\\)|\\【.*?】', '', doc)
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
    sentence = ' '.join(sentence)
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

def draw_cloud(mask, word_freq):
    '''
    Desc: 根据词频绘制词云图
    Args: mask 绘制图形形状
          word_freq 词频字典
    '''

    word_cloud = WordCloud(font_path='/usr/share/fonts/windows/MSYH.ttf', # 字体
                           # background_color='white',   # 背景颜色
                           width=1000,
                           height=600,
                           max_font_size=50,            # 字体大小
                           min_font_size=10,
                           mask=mask,                   # 背景图片
                           max_words=400,
                           scale = 1.8)
    word_cloud = word_cloud.fit_words(word_freq)
    word_cloud.to_file('wordcloud.png')
    plt.figure()
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()


# 自定义文本位置
text_file = '/home/kdd/nlp/业务约定书.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'

# 自定义词云图背景
mask = plt.imread('duck.jpeg')


if __name__ == '__main__':

    # 获取文本和停用词
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    # print(doc)
    
    # 切分成句子
    sents = doc2sent(doc)  
    # print(sents)


    # 分词、统计词频
    words = segment_jieba(sents, stopWords) # 结巴分词
    word_freq = dict(nltk.FreqDist(nltk.tokenize.word_tokenize(' '.join(words)))) 
    # words = segment_ltp(sents, stopwords_file, cws_model_path) # LTP分词
    # print(words)

    # 绘制词云图
    draw_cloud(mask, word_freq)


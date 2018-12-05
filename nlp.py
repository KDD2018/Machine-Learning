#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
from jieba import posseg as psg
from jieba import analyse
import nltk
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator as Imag
import matplotlib.pyplot as plt
import docx


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
    Args: file_name 文档位置
    Returns: document(str) 文档字符串
    '''

    document = ''
    file = docx.Document(file_name)
    for paragraph in file.paragraphs:
        graph = paragraph.text.replace(' ', '')  # 除去段内空格
        document += graph + '\n'
    return document

def doc2sent(doc):
    '''
    Desc: 除去文档中的空行并组成字符串
    Args: doc 文档字符串
    Returns: sents(str) 句子字符串
    '''
    sentences = doc.split('\n')  # 切分成句子
    sents = ''
    for sent in sentences:
        if sent == '':
            continue
        sents += sent + '\n'
    return sents

def sent2word(sentence, stopwords):
    """
    Desc: 分词并除去停用词
    Args: sentence 待分词的文本内容
          stopwords 停用词
    Returns: word_str(list) 分词结果
    """

    jieba.load_userdict('/home/kdd/userdict.txt')
    segResult = jieba.lcut(sentence)
    word_str = ''
    for word in segResult:
        if word in stopwords:
            continue
        elif word.isspace():
            continue
        else:
            # newSent.append(word)
            word_str += word + ' '  
    return word_str

def pos_tag(word_str):
    # 词性标注
    word_pog = psg.cut(word_str)
    tag = {}
    for w in word_pog:
        # print(w.word, w.flag)
        tag[w.word] = w.flag
    return tag

def draw_cloud(mask, word_freq):
    '''
    Desc: 根据词频绘制词云图
    Args: mask 绘制图形形状
          word_freq 词频字典
    '''

    word_cloud = WordCloud(font_path='/usr/share/fonts/windows/msyh.ttf', # 字体
                           # background_color='white',   #背景颜色
                           width=1000,
                           height=600,
                           max_font_size=50,            #字体大小
                           min_font_size=10,
                           mask=mask,                   #背景图片
                           max_words=400,
                           scale = 1.8)
    word_cloud = word_cloud.fit_words(word_freq)
    word_cloud.to_file('wordcloud.png')
    plt.figure('Word Cloud')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.show()


# 自定义文本位置
text_file = '/home/share/业务约定书.docx'
stopwords_file = '/home/kdd/stop_words.txt'

# 自定义词云图背景
mask = plt.imread('heart.jpeg')



if __name__ == '__main__':

    # 获取文本和停用词
    # text = get_text(file_name=text_file)
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    # print(doc)
    
    # 切分成句子
    sents = doc2sent(doc)  
    # print(sents)

    # 分词、统计词频
    words = sent2word(sents, stopWords)
    word_freq = dict(nltk.FreqDist(words.split(' ')).most_common(25))
    # print(words)

    # oft = fd.most_common(200)l
    # word_freq = dict(oft)
    print(word_freq)

    # 绘制词云图
    # draw_cloud(mask, word_freq)
    # tag = pos_tag(words) # 词性标注
    # print(tag)

    # 关键词抽取
    topic = analyse.extract_tags(words, topK=25)
    print('基于TF-IDF算法: \n', topic)
    topic = analyse.textrank(words, topK=25)
    print('\n基于TextRank算法: \n', topic)







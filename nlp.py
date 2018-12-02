#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 获取文本数据
def get_text(fila_name):
    f = open(fila_name, 'r', encoding='utf-8')
    text = f.read()
    f.close()
    return text

# 分词并除去停用词
def sent2word(sentence, stopwords):
    """
    Segment a sentence to words
    Delete stopwords
    """
    segResult = jieba.lcut(sentence)
    newSent = []
    for word in segResult:
        if word in stopwords:
            continue
        elif word.isspace():
            continue
        else:
            newSent.append(word)
    return newSent

text = get_text('/home/kdd/nlp/beauti.txt')  # 获取文本数据
stopWords = get_text('/home/kdd/nlp/stop_words.txt')  # 停用词

# sents = nltk.word_tokenize(text)

# 分词
words = sent2word(text, stopWords)


# 统计词频
fd = nltk.FreqDist(words)
oft = fd.most_common(400)

# print(oft)


# 绘制词云图
wc = WordCloud(font_path='/usr/share/fonts/wps-office/simhei.ttf',
               # background_color='white',   #背景颜色
               width=1000,
               height=600,
               max_font_size=50,            #字体大小
               min_font_size=10,
               # mask=plt.imread('xin.jpg'),  #背景图片
               max_words=400)
wc.generate(text)
wc.to_file('luyun.png')

plt.figure('word cloud')
plt.imshow(wc)
plt.axis('off')
plt.show()
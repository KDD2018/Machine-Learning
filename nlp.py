#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
import nltk
from wordcloud import WordCloud

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

text = get_text('/home/share/TEXT/天下第九.txt')  # 获取文本数据
stopWords = get_text('/home/kdd/stop_words.txt')  # 停用词

# sents = nltk.word_tokenize(text)
# words = []
# for sent in sents:
#     words.append(sent2word(text, stopWords))
words = sent2word(text, stopWords)

# print(len(set(words)))
# print(stopWords)
# print(sents)

fd = nltk.FreqDist(words)
oft = fd.most_common(400)
# cf = fd.most_common(20)
# print(oft)
dd = dict(oft)
wc = WordCloud()
wc.generate(dd)
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
import nltk
import matplotlib.pyplot as plt
import docx
import os
import re
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer as NER




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

def pos_tag(word_list, pos_model_path):
    '''
    Desc: LTP词性标注
    Args: word_list 分词结果
          pos_model_path 词性标注模型
    Returns: word_tag 词性词典
    '''

    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(word_list)
    postagger.release()
    word_tag = dict(zip(word_list, list(postags)))
    return word_tag

def recognize(word_tag, ner_model_path):
    '''
    Desc: 命名实体识别
    Args: word_tag(dict) 词性词典
          ner_model_path 实体识别模型
    Returns: ner_tag 命名实体标签 
    '''
    
    recog = NER()
    recog.load(ner_model_path)
    ner_tag = recog.recognize(list(word_tag.keys()), list(word_tag.values()))
    recog.release()
    # ner_tag = dict(zip(list(word_tag.keys()), list(ner_tag)))
    return ner_tag


# 自定义文本位置
text_file = '/home/kdd/nlp/业务约定书.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'

# ltp模型路径
LTP_DATA_DIR = '/home/kdd/nlp/ltp_data_v3.4.0/' 
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`



if __name__ == '__main__':

    # 获取文本和停用词
    # text = get_text(file_name=text_file)
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    # print(doc)
    
    # 切分成句子
    sents = doc2sent(doc)  
    # print(sents)

    # 信息抽取
    for sent in sents:
        # print(sent)
        words = segment_jieba(sent, stopWords)
        # print(words)
        word_tag = pos_tag(words, pos_model_path)
        print(word_tag)
        ner_tag = recognize(word_tag, ner_model_path)
        # print(list(ner_tag))
        # NE_list = set()
        # for i in range(len(ner_tag)):
        #     if ner_tag[i][0] == 'S' or ner_tag[i][0] == 'B':
        #         j = i
        #         if ner_tag[j][0] == 'B':
        #             while ner_tag[j][0] != 'E':
        #                 j += 1
        #                 print('='*30)
        #             e = ''.join(words[i:j+1])
        #             NE_list.add(e)
        #         else:
        #             e = words[j]
        #             NE_list.add(e)
        # print(NE_list)
        # arcs = parser(word_tag, par_model_path)
        # arc_str = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        # arcs_dict[tuple(words)] = arc_str
        # labeller(word_tag, arcs, srl_model_path) # 语义角色标注
    # print(arcs_dict)



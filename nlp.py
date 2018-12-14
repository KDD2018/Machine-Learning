#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
from jieba import posseg as psg
from jieba import analyse
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import docx
import os
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer as NER
from pyltp import Parser
from pyltp import SementicRoleLabeller
from snownlp import normal
from snownlp.summary import textrank


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
    Returns: sents(list) 句子列表
    '''
    
    sentences = SentenceSplitter.split(doc)  # 切分成句子
    sents = []
    # sents = ''
    for sent in sentences:
        if sent == '':
            continue
        # sents += sent + '\n'
        sents.append(sent)
    return sents

def segment_jieba(sentence, stopwords):
    """
    Desc: 分词并除去停用词
    Args: sentence 待分词的文本内容
          stopwords 停用词
    Returns: words 分词结果
    """

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

    # word_pog = psg.cut(word_str)
    # tag = {}
    # for w in word_pog:
    #     # print(w.word, w.flag)
    #     tag[w.word] = w.flag
    # return tag

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
    ner_tag = dict(zip(list(word_tag.keys()), list(ner_tag)))
    return ner_tag

def parser(word_tag, par_model_path):
    '''
    Desc: 依存句法分析
    Args: word_tag(dict) 词性词典
          par_model_path 依存句法分析模型
    Returns: acrs 依存关系
    '''

    parser = Parser()
    parser.load(par_model_path)
    arcs = parser.parse(list(word_tag.keys()), list(word_tag.values()))
    parser.release()
    return arcs

def labeller(word_tag, arcs, srl_model_path):
    '''
    Desc: 语义角色标注
    Args: word_tag(dict) 词性词典
          arcs 依存关系
          srl_model_path 语义角色标注模型
    '''

    labeller = SementicRoleLabeller()
    labeller.load(srl_model_path)
    roles = labeller.label(list(word_tag.keys()), list(word_tag.values()), arcs)
    for role in roles:
        print(role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
    labeller.release()  


# 自定义文本位置
text_file = '/home/kdd/nlp/评估报告.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'

# ltp模型路径
LTP_DATA_DIR = '/home/kdd/nlp/ltp_data_v3.4.0/' 
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`

# 自定义词云图背景
mask = plt.imread('duck.jpeg')



if __name__ == '__main__':

    # 获取文本和停用词
    # text = get_text(file_name=text_file)
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    print(doc)
    
    # 切分成句子
    # sents = SentenceSplitter.split(doc)
    sents = doc2sent(doc)  
    # print(sents)

    
    # 分词、统计词频
    words = segment_jieba(sents, stopWords) # 结巴分词
    word_freq = dict(nltk.FreqDist(nltk.tokenize.word_tokenize(' '.join(words)))) 
    # words = segment_ltp(sents, stopwords_file, cws_model_path) # LTP分词
    # print(word_freq)

    # 绘制词云图
    draw_cloud(mask, word_freq)


    # 词性标注
    word_tag = pos_tag(words, pos_model_path) 
    print(word_tag)

    # 命名实体识别
    ner_tag = recognize(word_tag, ner_model_path)
    print(ner_tag)
    

    
    # 依存句法分析
    arcs_dict = {}
    for sent in sents:
        # print(sent)
        word_list = segment_jieba(sent, stopWords)
        # print(word_list)
        word_tag = pos_tag(word_list, pos_model_path)
        # print(word_tag)
        arcs = parser(word_tag, par_model_path)
        arc_str = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        arcs_dict[tuple(word_list)] = arc_str
        labeller(word_tag, arcs, srl_model_path) # 语义角色标注
    print(arcs_dict)
    

    
    # 摘要提取
    # sent_str = ' '.join(sents)
    # print(sents)
    word_list = []
    for sent in sents:
        words = segment_jieba(sent, stopWords)
        word_list.append(words)
    # print(word_list)

    rank = textrank.TextRank(word_list)
    rank.solve()
    for index in rank.top_index(5):
        print(sents[index])
    

    # keyword_rank = textrank.KeywordTextRank(word_list)
    # keyword_rank.solve()
    # for w in keyword_rank.top_index(10):
    #     print(w)

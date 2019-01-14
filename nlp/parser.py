#!/usr/bin/python3
# -*- coding: utf-8 -*-

import jieba
import docx
import os
import re
import pkuseg
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer as NER
from pyltp import Parser
from pyltp import SementicRoleLabeller


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

def segment_pku(sentence, stopwords):
    '''
    Desc: pkuseg分词
    Args: sentence 待分词文本
          stopwords 停用词
          model_name 分词模型
    Returns: words 分词列表
    '''

    user_dict = ['资产评估', '业务约定书', '北京四方清能电气电子有限公司', '凯晨世贸中心', '委托方', '受托方', '中联资产评估集团有限公司', '北京银行',
                 '中国资产评估协会', '北京四方继保自动化股份有限公司', 'XXXX有限公司', '复兴门内大街28号', '约定书', '专家国际公馆', '后屯路26号']
    seg = pkuseg.pkuseg(model_name='ctb8', user_dict=user_dict)
    segResult = seg.cut(sentence)
    words = []
    for word in segResult:
        if word in stopwords:
            continue
        elif word.isspace():
            continue
        else:
            words.append(word)
    return words

def pos_tag(words, pos_model_path):
    '''
    Desc: LTP词性标注
    Args: words(list) 分词结果
          pos_model_path 词性标注模型
    Returns: word_tag 词性词典
    '''

    postagger = Postagger()
    postagger.load(pos_model_path)
    postags = postagger.postag(words)
    postagger.release()
    # word_tag = dict(zip(words, list(postags)))
    return postags

def recognize(words, postags, ner_model_path):
    '''
    Desc: 命名实体识别
    Args: word_tag(dict) 词性词典
          ner_model_path 实体识别模型
    Returns: ner_tag 命名实体标签 
    '''
    
    recog = NER()
    recog.load(ner_model_path)
    ner_tag = recog.recognize(words, postags)
    recog.release()
    # ner_tag = dict(zip(list(word_tag.keys()), list(ner_tag)))
    return ner_tag

def parser(words, postags, par_model_path):
    '''
    Desc: 依存句法分析
    Args: word_tag(dict) 词性词典
          par_model_path 依存句法分析模型
    Returns: acrs 依存关系
    '''

    parser = Parser()
    parser.load(par_model_path)
    arcs = parser.parse(words, postags)
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

def build_parse_child_dict(words, postags, arcs):
    """
    Desc: 为句子中的每个词语维护一个保存句法依存儿子节点的字典
    Args:
        words: 分词列表
        postags: 词性列表
        arcs: 句法依存列表
    """
    child_dict_list = []
    for index in range(len(words)):
        child_dict = dict()
        for arc_index in range(len(arcs)):
            if arcs[arc_index].head == index + 1:
                if arcs[arc_index].relation in child_dict:
                    child_dict[arcs[arc_index].relation].append(arc_index)
                else:
                    child_dict[arcs[arc_index].relation] = []
                    child_dict[arcs[arc_index].relation].append(arc_index)

        child_dict_list.append(child_dict)
    return child_dict_list

def complete_e(words, postags, child_dict_list, word_index):
    """
    完善识别的部分实体
    """
    child_dict = child_dict_list[word_index]
    prefix = ''
    if 'ATT' in child_dict:
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    
    postfix = ''
    if postags[word_index] == 'v':
        if 'VOB' in child_dict:
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if 'SBV' in child_dict:
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix

def is_good(e, NE_list, sentence):
    """
    判断e是否为命名实体
    """
    if e not in sentence:
        return False

    words_e = segment_jieba(e, stopWords)
    postags_e = pos_tag(words_e, pos_model_path)
    if e in NE_list:
        return True
    else:
        NE_count = 0
        for i in range(len(words_e)):
            if words_e[i] in NE_list:
                NE_count += 1
            if postags_e[i] == 'v':
                return False
        if NE_count >= len(words_e)-NE_count:
            return True
    return False

def fact_triple_extract(sentence, out_file, corpus_file):
    """
    对于给定的句子进行事实三元组抽取
    Args:
        sentence: 要处理的语句
    """

    words = segment_jieba(sentence, stopWords)
    # print "\t".join(words)
    postags = pos_tag(words, pos_model_path)
    netags = recognize(words, postags, ner_model_path)
    arcs = parser(words, postags, par_model_path)
    # print "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)


    NE_list = set()
    for i in range(len(netags)):
        if netags[i][0] == 'S' or netags[i][0] == 'B':
            j = i
            if netags[j][0] == 'B':
                while netags[j][0] != 'E':
                    j += 1
                e = ''.join(words[i:j + 1])
                NE_list.add(e)
            else:
                e = words[j]
                NE_list.add(e)

    corpus_flag = False
    child_dict_list = build_parse_child_dict(words, postags, arcs)
    for index in range(len(postags)):
        # 抽取以谓词为中心的事实三元组
        if postags[index] == 'v':
            child_dict = child_dict_list[index]
            # 主谓宾
            if 'SBV' in child_dict and 'VOB' in child_dict:
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                r = words[index]
                e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                # if e1 in NE_list or e2 in NE_list:
                if is_good(e1, NE_list, sentence) and is_good(e2, NE_list, sentence):
                    out_file.write("主语谓语宾语关系\t(%s, %s, %s)\n" % (e1, r, e2))
                    out_file.flush()
                    if not corpus_flag:
                        corpus_file.write(sentence)
                        corpus_flag = True
                    e1_start = (sentence.decode('utf-8')).index((e1.decode('utf-8')))
                    e1_end = e1_start + len(e1.decode('utf-8')) - 1
                    r_start = (sentence.decode('utf-8')).index((r.decode('utf-8')))
                    r_end = r_start + len(r.decode('utf-8')) - 1
                    e2_start = (sentence.decode('utf-8')).index((e2.decode('utf-8')))
                    e2_end = e2_start + len(e2.decode('utf-8')) - 1
                    corpus_file.write("$$3==%s/%d-%d==%s/%d-%d==%s/%d-%d" % (
                    e1, e1_start, e1_end, r, r_start, r_end, e2, e2_start, e2_end))
                    corpus_file.flush()
            # 定语后置，动宾关系
            if arcs[index].relation == 'ATT':
                if 'VOB' in child_dict:
                    e1 = complete_e(words, postags, child_dict_list, arcs[index].head - 1)
                    r = words[index]
                    e2 = complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
                    temp_string = r + e2
                    if temp_string == e1[:len(temp_string)]:
                        e1 = e1[len(temp_string):]
                    # if temp_string not in e1 and (e1 in NE_list or e2 in NE_list):
                    if temp_string not in e1 and is_good(e1, NE_list, sentence) and is_good(e2, NE_list, sentence):
                        out_file.write("定语后置动宾关系\t(%s, %s, %s)\n" % (e1, r, e2))
                        out_file.flush()
                        if not corpus_flag:
                            corpus_file.write(sentence)
                            corpus_flag = True
                        e1_start = (sentence.decode('utf-8')).index((e1.decode('utf-8')))
                        e1_end = e1_start + len(e1.decode('utf-8')) - 1
                        r_start = (sentence.decode('utf-8')).index((r.decode('utf-8')))
                        r_end = r_start + len(r.decode('utf-8')) - 1
                        e2_start = (sentence.decode('utf-8')).index((e2.decode('utf-8')))
                        e2_end = e2_start + len(e2.decode('utf-8')) - 1
                        corpus_file.write("$$3==%s/%d-%d==%s/%d-%d==%s/%d-%d" % (
                        e1, e1_start, e1_end, r, r_start, r_end, e2, e2_start, e2_end))
                        corpus_file.flush()
            # 含有介宾关系的主谓动补关系
            if 'SBV' in child_dict and 'CMP' in child_dict:
                # e1 = words[child_dict['SBV'][0]]
                e1 = complete_e(words, postags, child_dict_list, child_dict['SBV'][0])
                cmp_index = child_dict['CMP'][0]
                r = words[index] + words[cmp_index]
                if child_dict_list[cmp_index].has_key('POB'):
                    e2 = complete_e(words, postags, child_dict_list, child_dict_list[cmp_index]['POB'][0])
                    # if e1 in NE_list or e2 in NE_list:
                    if is_good(e1, NE_list, sentence) and is_good(e2, NE_list, sentence):
                        out_file.write("介宾关系主谓动补\t(%s, %s, %s)\n" % (e1, r, e2))
                        out_file.flush()
                        if not corpus_flag:
                            corpus_file.write(sentence)
                            corpus_flag = True
                        e1_start = (sentence.decode('utf-8')).index((e1.decode('utf-8')))
                        e1_end = e1_start + len(e1.decode('utf-8')) - 1
                        r_start = (sentence.decode('utf-8')).index((r.decode('utf-8')))
                        r_end = r_start + len(r.decode('utf-8')) - 1
                        e2_start = (sentence.decode('utf-8')).index((e2.decode('utf-8')))
                        e2_end = e2_start + len(e2.decode('utf-8')) - 1
                        corpus_file.write("$$3==%s/%d-%d==%s/%d-%d==%s/%d-%d" % (
                        e1, e1_start, e1_end, r, r_start, r_end, e2, e2_start, e2_end))
                        corpus_file.flush()
        # 尝试抽取命名实体有关的三元组
        if netags[index][0] == 'S' or netags[index][0] == 'B':
            ni = index
            if netags[ni][0] == 'B':
                while netags[ni][0] != 'E':
                    ni += 1
                e1 = ''.join(words[index:ni + 1])
            else:
                e1 = words[ni]
            if arcs[ni].relation == 'ATT' and postags[arcs[ni].head - 1] == 'n' and netags[arcs[ni].head - 1] == 'O':
                r = complete_e(words, postags, child_dict_list, arcs[ni].head - 1)
                if e1 in r:
                    r = r[(r.index(e1) + len(e1)):]
                if arcs[arcs[ni].head - 1].relation == 'ATT' and netags[arcs[arcs[ni].head - 1].head - 1] != 'O':
                    e2 = complete_e(words, postags, child_dict_list, arcs[arcs[ni].head - 1].head - 1)
                    mi = arcs[arcs[ni].head - 1].head - 1
                    li = mi
                    if netags[mi][0] == 'B':
                        while netags[mi][0] != 'E':
                            mi += 1
                        e = ''.join(words[li + 1:mi + 1])
                        e2 += e
                    if r in e2:
                        e2 = e2[(e2.index(r) + len(r)):]
                    if is_good(e1, NE_list, sentence) and is_good(e2, NE_list, sentence):
                        out_file.write("人名//地名//机构\t(%s, %s, %s)\n" % (e1, r, e2))
                        out_file.flush()
                        if not corpus_flag:
                            corpus_file.write(sentence)
                            corpus_flag = True
                        e1_start = (sentence.decode('utf-8')).index((e1.decode('utf-8')))
                        e1_end = e1_start + len(e1.decode('utf-8')) - 1
                        r_start = (sentence.decode('utf-8')).index((r.decode('utf-8')))
                        r_end = r_start + len(r.decode('utf-8')) - 1
                        e2_start = (sentence.decode('utf-8')).index((e2.decode('utf-8')))
                        e2_end = e2_start + len(e2.decode('utf-8')) - 1
                        corpus_file.write("$$3==%s/%d-%d==%s/%d-%d==%s/%d-%d" % (
                        e1, e1_start, e1_end, r, r_start, r_end, e2, e2_start, e2_end))
                        corpus_file.flush()
    return corpus_flag

def extact_start(sents, out_file_path, corpus_file_path):
    '''
    Desc: 抽取总控开关
    Args: out_file 输出结果文件 
          corpus_file 语料文件
     
    '''
    out_file = open(out_file_path)
    corpus_file = open(corpus_file_path)

    info_dict = dict()
    info_dict['title'] = sents[0]
    for sent in sents:
        # print(sent)
        if '：' in sent and sent.split('：')[1] != '':
            # info_dict = dict()
            info_dict[sent.split('：')[0]] = sent.split('：')[1]
        else:
            try:
                flag = fact_triple_extract(sent, out_file, corpus_file)
                out_file.flush()
                if flag:
                    corpus_file.write('\n')
                    corpus_file.flush()
            except:
                pass
    out_file.close()
    corpus_file.close()

    # print(info_dict)



# 自定义文本位置
text_file = '/home/kdd/nlp/业务约定书.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'
out_file_path = '/home/kdd/nlp/out_file.txt'
corpus_file_path = '/home/kdd/nlp/corpus.txt'

# ltp模型路径s
LTP_DATA_DIR = '/home/kdd/nlp/ltp_data_v3.4.0/' 
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`


if __name__ == '__main__':

    # 获取文本和停用词
    # text = get_text(file_name=text_file)
    doc = get_doc(text_file)
    stopWords = get_text(file_name=stopwords_file)
    # print(doc)
    
    # 切分成句子
    sents = doc2sent(doc)
    # print(sents)

    # 依存句法分析
    fact_triple_extract(sents, out_file_path, corpus_file_path)
    '''
    arcs_dict = dict()
    info_dict = dict()
    info_dict['title'] = sents[0]
    for sent in sents:
        # print(sent)
        if '：' in sent and sent.split('：')[1] != '':
            # info_dict = dict()
            info_dict[sent.split('：')[0]] = sent.split('：')[1]
        else:
            # words = segment_jieba(sent, stopWords)
            # # words = segment_pku(sent, stopWords)
            # # print(words)
            # word_tag = pos_tag(words, pos_model_path)
            # # print(word_tag)
            # ner_tag = recognize(word_tag, ner_model_path)
            # arcs = parser(word_tag, par_model_path)
            # arc_str = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
            # arcs_dict[tuple(words)] = arc_str
            # # labeller(word_tag, arcs, srl_model_path) # 语义角色标注
            # child_dict_list = build_parse_child_dict(words, word_tag.values(), arcs)
            cor = fact_triple_extract(sent, out_file_name, corpus_file_name)
    '''









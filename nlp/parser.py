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
import re
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
    word_tag = dict(zip(words, list(postags)))
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
        #if child_dict.has_key('SBV'):
        #    print words[index],child_dict['SBV']
        child_dict_list.append(child_dict)
    return child_dict_list

def complete_e(words, postags, child_dict_list, word_index):
    """
    完善识别的部分实体
    """
    child_dict = child_dict_list[word_index]
    prefix = ''
    if child_dict.has_key('ATT'):
        for i in range(len(child_dict['ATT'])):
            prefix += complete_e(words, postags, child_dict_list, child_dict['ATT'][i])
    
    postfix = ''
    if postags[word_index] == 'v':
        if child_dict.has_key('VOB'):
            postfix += complete_e(words, postags, child_dict_list, child_dict['VOB'][0])
        if child_dict.has_key('SBV'):
            prefix = complete_e(words, postags, child_dict_list, child_dict['SBV'][0]) + prefix

    return prefix + words[word_index] + postfix

def is_good(e, NE_list, sentence):
    """
    判断e是否为命名实体
    """
    if e not in sentence:
        return False

    words_e = segmentor.segment(e)
    postags_e = postagger.postag(words_e)
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

def extraction_start_from_xml(in_file_name):
    """
    提取文本中的text标签的内容，进行实体关系三元组抽取
    """
    docs_root = etree.parse(in_file_name).getroot()
    out_file = open(in_file_name+'.triple.txt', 'w')
    out_docs_root = etree.Element("docs")
    sentence_count = 0
    find_flag = False
    for each_doc in docs_root:  # 遍历每个doc
        out_doc_element = etree.SubElement(out_docs_root, "doc")
        out_doc_element.attrib["name"] = each_doc.attrib["name"]
        out_doc_element.attrib["url"] = each_doc.attrib["url"]
        out_doc_element.attrib["id"] = each_doc.attrib["id"]
        out_doc_element.attrib["baike_id"] = each_doc.attrib["baike_id"]
        out_doc_element.attrib["time"] = each_doc.attrib["time"]
        for each_par in each_doc:
            out_par_element = etree.SubElement(out_doc_element, "par")
            for element in each_par:
                if element.tag == "text":  
                    text = element.text.encode('utf-8')
                    text = text.replace("。","。\n").replace("！","！\n").replace("？","？\n")
                    sentences = text.split("\n")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence == '':
                            continue
                        sentence_count += 1
                        if sentence_count%1000 == 0:
                            print(sentence_count,"sentences done.")
                        u_sentence = sentence.decode('utf-8')
                        out_sentence_element = etree.SubElement(out_par_element, "sentence")
                        out_s_text_element = etree.SubElement(out_sentence_element, "s_text")
                        out_s_text_element.text = u_sentence
                        try:
                            find_flag = fact_triple_extract(sentence, out_file, out_sentence_element)
                            if find_flag == False:
                                out_sentence_element.xpath("..")[0].remove(out_sentence_element)
                            out_file.flush()
                        except:
                            pass
            if find_flag == False:
                out_par_element.xpath("..")[0].remove(out_par_element)
        if find_flag == False:
            out_doc_element.xpath("..")[0].remove(out_doc_element)
    tree = etree.ElementTree(out_docs_root)
    tree.write(in_file_name+".triple.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')

def extraction_start_from_xml(in_file_name):
    """
    提取文本中的text标签的内容，进行实体关系三元组抽取
    """
    docs_root = etree.parse(in_file_name).getroot()
    out_file = open(in_file_name+'.triple.txt', 'w')
    out_docs_root = etree.Element("docs")
    sentence_count = 0
    find_flag = False
    for each_doc in docs_root:  # 遍历每个doc
        out_doc_element = etree.SubElement(out_docs_root, "doc")
        out_doc_element.attrib["name"] = each_doc.attrib["name"]
        out_doc_element.attrib["url"] = each_doc.attrib["url"]
        out_doc_element.attrib["id"] = each_doc.attrib["id"]
        out_doc_element.attrib["baike_id"] = each_doc.attrib["baike_id"]
        out_doc_element.attrib["time"] = each_doc.attrib["time"]
        for each_par in each_doc:
            out_par_element = etree.SubElement(out_doc_element, "par")
            for element in each_par:
                if element.tag == "text":  
                    text = element.text.encode('utf-8')
                    text = text.replace("。","。\n").replace("！","！\n").replace("？","？\n")
                    sentences = text.split("\n")
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence == '':
                            continue
                        sentence_count += 1
                        if sentence_count%1000 == 0:
                            print(sentence_count,"sentences done.")
                        u_sentence = sentence.decode('utf-8')
                        out_sentence_element = etree.SubElement(out_par_element, "sentence")
                        out_s_text_element = etree.SubElement(out_sentence_element, "s_text")
                        out_s_text_element.text = u_sentence
                        try:
                            find_flag = fact_triple_extract(sentence, out_file, out_sentence_element)
                            if find_flag == False:
                                out_sentence_element.xpath("..")[0].remove(out_sentence_element)
                            out_file.flush()
                        except:
                            pass
            if find_flag == False:
                out_par_element.xpath("..")[0].remove(out_par_element)
        if find_flag == False:
            out_doc_element.xpath("..")[0].remove(out_doc_element)
    tree = etree.ElementTree(out_docs_root)
    tree.write(in_file_name+".triple.xml", pretty_print=True, xml_declaration=True, encoding='utf-8')


# 自定义文本位置
text_file = '/home/kdd/nlp/业务约定书.docx'
stopwords_file = '/home/kdd/nlp/stop_words.txt'

# ltp模型路径
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
    arcs_dict = {}
    for sent in sents:
        # print(sent)
        words = segment_jieba(sent, stopWords)
        # print(words)
        word_tag = pos_tag(words, pos_model_path)
        # print(word_tag)
        ner_tag = recognize(word_tag, ner_model_path)
        arcs = parser(word_tag, par_model_path)
        arc_str = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        arcs_dict[tuple(words)] = arc_str
        # labeller(word_tag, arcs, srl_model_path) # 语义角色标注
        child_dict_list = build_parse_child_dict(words, word_tag.values(), arcs)
        print(child_dict_list)


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




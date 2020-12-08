# _*_ coding: utf-8 _*_
# @CreateTime : 2020/10/31 18:23 
# @Author : zkz 
# @File : segment_THUCNews.py


import jieba
from os import listdir
import io
import math
import re
import xml.etree.ElementTree as ET


# print(len(files))
# print(files)
def get_stop_words():
    f = open('../data/stop_words.txt', encoding = 'utf-8')
    stop_words = f.read()
    stop_words = set(stop_words.split('\n'))
    #print(stop_words)

    # stopwords = []
    # with open(path, "r", encoding='utf8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         stopwords.append(line.strip())
    return stop_words

def get_word_crups():
    stop_words = get_stop_words()
    path = './data/news/'
    files = listdir(path)
    crops_file = open('./data/crops.txt', 'w', encoding='utf8')
    for i in files:
        root = ET.parse(path + i).getroot()
        body = root.find('body').text
        sentences = re.split('[\n\t]', body)
        for sentence in sentences:
            word_crops = []
            if sentence == '':
                continue
            words = list(jieba.lcut(sentence, cut_all = False))
            #temp = []
            for word in words:
                if word not in stop_words:
                    word_crops.append(word)
            #word_crops.append(temp)
            crops_file.write(" ".join(word_crops) + '\n')
        #break

def get_doc(docid):
    stop_words = get_stop_words()
    path = '../data/news/'
    #files = listdir(path)
    #crops_file = open('./data/crops.txt', 'w', encoding='utf8')
    #for i in files:
    root = ET.parse(path + str(docid) + '.xml').getroot()
    body = root.find('body').text
    sentences = re.split('[\n\t]', body)
    word_crops = []
    for sentence in sentences:
        if sentence == '':
            continue
        words = list(jieba.lcut(sentence, cut_all = False))
        for word in words:
            if word not in stop_words and word not in word_crops:
                word_crops.append(word)

    return word_crops


if __name__ == "__main__":
    get_word_crups()
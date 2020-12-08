# _*_ coding: utf-8 _*_
# @CreateTime : 2020/10/31 20:47 
# @Author : zkz 
# @File : word2vec.py.py

from gensim.models import word2vec
import multiprocessing

from segment_THUCNews import get_word_crups

def train_wordVectors(sentences, embedding_size=128, window=5, min_count=5):
    '''
    :param sentences: sentences可以是LineSentence或者PathLineSentences读取的文件对象，也可以是
                    The `sentences` iterable can be simply a list of lists of tokens,如lists=[['我','是','中国','人'],['我','的','家乡','在','广东']]
    :param embedding_size: 词嵌入大小
    :param window: 窗口
    :param min_count:Ignores all words with total frequency lower than this.
    :return: w2vModel
    '''
    w2vModel = word2vec.Word2Vec(sentences, size=embedding_size, window=window, min_count=min_count,
                                 workers=multiprocessing.cpu_count())
    return w2vModel


def save_wordVectors(w2vModel, word2vec_path):
    w2vModel.save(word2vec_path)


def load_wordVectors(word2vec_path):
    w2vModel = word2vec.Word2Vec.load(word2vec_path)
    return w2vModel


if __name__ == '__main__':
    # [1]若只有一个文件，使用LineSentence读取文件
    # segment_path='./data/crops.txt'
    # sentences = word2vec.LineSentence(segment_path)
    #
    # # 简单的训练
    # model = word2vec.Word2Vec(sentences, hs=1, min_count=1, window=3, size=100)
    # print(model.wv.similarity('郑州', '河南'))
    # print(model.wv.similarity('北京', '郑州'))
    word2vec_path = './models/word2Vec.model'
    # save_wordVectors(model, word2vec_path)
    model = load_wordVectors(word2vec_path)


    print(model['郑州'], type(model['郑州']), model['郑州'].shape)
    term = '郑州'
    print(model[term])
    print(model.wv.vocab.keys())

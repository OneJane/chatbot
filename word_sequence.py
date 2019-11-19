# -*- coding:utf-8 -*-

import numpy as np


# 句子转为向量
class WordSequence(object):
    PAD_TAG = '<pad>' # 填充标签
    UNK_TAG = '<unk>' # 未知标签
    START_TAG = '<s>'
    END_TAG = '</S>'

    PAD = 0
    UNK = 1
    START = 2
    END = 3

    def __init__(self):
        self.fited = False # 是否训练
        self.dict = {WordSequence.PAD_TAG: WordSequence.PAD, WordSequence.UNK_TAG: WordSequence.UNK,
                     WordSequence.START_TAG: WordSequence.START, WordSequence.END_TAG: WordSequence.END} # 初始化字典

    # 词转为index
    def to_index(self, word):
        assert self.fited, 'WordSequence尚未进行fit操作'
        if word in self.dict:
            return self.dict[word]  # 词在字典中的位置
        return WordSequence.UNK

    # index转为词
    def to_word(self, index):
        assert self.fited, 'WordSequence尚未进行fit操作'
        for k, v in self.dict.items():
            if v == index:
                return k
        return WordSequence.UNK_TAG

    # 获取长度
    def size(self):
        assert self.fited, 'WordSequence尚未进行fit操作'
        return len(self.dict) + 1

    def __len__(self):
        return self.size()

    # 拟合 训练
    def fit(self, sentences, min_count=5, max_count=None, max_features=None):
        assert not self.fited, 'WordSequence只能fit一次'

        count = {}
        # 统计句子
        for sentence in sentences:
            arr = list(sentence)
            for a in arr:
                if a not in count:
                    count[a] = 0  # 没有被统计
                count[a] += 1
        if min_count is not None:
            # 大于最小值
            count = {k: v for k, v in count.items() if v >= min_count}
        if max_count is not None:
            # 小于最大值
            count = {k: v for k, v in count.items() if v <= max_count}

        self.dict = {WordSequence.PAD_TAG: WordSequence.PAD, WordSequence.UNK_TAG: WordSequence.UNK,
                     WordSequence.START_TAG: WordSequence.START, WordSequence.END_TAG: WordSequence.END}

        if isinstance(max_features, int):
            count = sorted(list(count.items()), key=lambda x: x[1])
            if max_features is not None and len(count) > max_features:
                count = count[-int(max_features):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)

        self.fited = True

    # 句子转为向量
    def transform(self, sentence, max_len=None):
        assert self.fited, 'WordSequence尚未进行fit操作'

        if max_len is not None:
            r = [self.PAD] * max_len
        else:
            r = [self.PAD] * len(sentence)

        for index, a in enumerate(sentence):
            if max_len is not None and index >= len(r):
                break
            r[index] = self.to_index(a)

        return np.array(r)

    # 向量转为句子
    def inverse_transform(self, indices, ignore_pad=False, ignore_unk=False, ignore_start=False, ignore_end=False):
        ret = []
        for i in indices:
            word = self.to_word(i)
            if word == WordSequence.PAD_TAG and ignore_pad:
                continue
            if word == WordSequence.UNK_TAG and ignore_unk:
                continue
            if word == WordSequence.START_TAG and ignore_start:
                continue
            if word == WordSequence.END_TAG and ignore_end:
                continue

            ret.append(word)

        return ret


def test():
    ws = WordSequence()
    ws.fit([['你', '好', '啊'], ['你', '好', '哦'], ])

    indice = ws.transform(['我', '们', '好'])
    print(indice)

    back = ws.inverse_transform(indice)
    print(back)


if __name__ == '__main__':
    test()

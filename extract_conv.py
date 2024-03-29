# -*- coding:utf-8 -*-

import re
import pickle
import sys
from tqdm import tqdm


# 处理预料 生成问答数据模型
# 分割
def make_split(line):
    # 你好？我是wj... -> 你好我是wj
    if re.match(r'.*([，···?!\.,!？])$', ''.join(line)):
        return []

    return [', ']


# 判断是否是好句子 句子中包含2个英文和数字以上则不是好句子
def good_line(line):
    if len(re.findall(r'[a-zA-Z0-9]', ''.join(line))) > 2:
        return False
    return True


# 英文字符替换为中文 你好........ -> 你好...
def regular(sen):
    sen = re.sub(r'\.{3,100}', '···', sen)
    sen = re.sub(r'···{2,100}', '···', sen)
    sen = re.sub(r'[,]{1,100}', '，', sen)
    sen = re.sub(r'[\.]{1,100}', '。', sen)
    sen = re.sub(r'[\?]{1,100}', '？', sen)
    sen = re.sub(r'[!]{1,100}', '！', sen)

    return sen


def main(limit=20, x_limit=3, y_limit=6):
    from word_sequence import WordSequence

    print('extract lines')
    fp = open('dataset/dgk_shooter_min.conv', 'r', errors='ignore', encoding='utf-8')
    groups = []
    group = []

    # tqdm 进度条库
    for line in tqdm(fp):
        if line.startswith('M '):
            line = line.replace('\n', '')
            if '/' in line:
                line = line[2:].split('/')
            else:
                line = list(line[2:])
            line = line[:-1]
            # 每行去除加入大数组里
            group.append(list(regular(''.join(line))))
        else:
            if group:
                groups.append(group)
                group = []
    if group:
        groups.append(group)
        group = []

    print('extract group')

    x_data = []
    y_data = []

    for group in tqdm(groups):
        for i, line in enumerate(group):  # M i;畹/华/吾/侄/ line

            last_line = None           # last_line next_line next_next_line问答操作
            if i > 0:                   # group至少两行0 1
                last_line = group[i-1]  # 获取最后一行
                if not good_line(last_line):
                    last_line = None
            next_line = None
            if i < len(group) - 1:
                next_line = group[i+1]
                if not good_line(next_line):
                    next_line = None

            next_next_line = None
            if i < len(group) - 2:
                next_next_line = group[i + 2]
                if not good_line(next_next_line):
                    next_next_line = None

            if next_line:      # 第一行数据存在下一行
                x_data.append(line)  # x赋第一行
                y_data.append(next_line)  # y赋第二行
            if last_line and next_line:
                x_data.append(last_line + make_split(last_line) + line)
                y_data.append(next_line)
            if next_line and next_next_line:
                x_data.append(line)
                y_data.append(next_line + make_split(next_line) + next_next_line)

    print(len(x_data), len(y_data))

    # 构建问答
    for ask, answer in zip(x_data[:20], y_data[:20]):
        print(''.join(ask))
        print(''.join(answer))
        print('-'*20)

    # 生成pkl文件备用
    data = list(zip(x_data, y_data))
    data = [
        (x, y) for x, y in data if limit > len(x) >= x_limit and limit > len(y) >= y_limit
    ]

    x_data, y_data = zip(*data)
    ws_input = WordSequence()
    ws_input.fit(x_data + y_data)
    print('dump')
    pickle.dump(
        (x_data, y_data), open('chatbot.pkl', 'wb'))
    pickle.dump(ws_input, open('ws.pkl', 'wb'))
    print('done')


if __name__ == '__main__':
    main()

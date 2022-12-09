# -*- coding: utf-8 -*-

"""
# @File       : init.py
# @Time       : 2022/11/30
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 初始化文件，用于设置文件目录
"""

import os
import pandas as pd

# ? 文件目录设置
SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果文件目录
TMP_DIR = os.path.join(BASE_DIR, 'tmp')  # 临时文件目录


def get_filename(filename: str) -> str:
    """由文件名获取文件绝对路径

    Args:
        filename (str): 文件名，不含目录

    Returns:
        str: 包含路径的文件名
    """
    return os.path.join(DATA_DIR, filename)


def get_utility_mat():
    # ? 获取用户-电影评分效应矩阵
    train_file = open(get_filename('train_set.csv'), 'r', encoding='utf-8')
    # * train_file: 存放每位用户评论的电影和评分
    # * train_file 是一个嵌套字典
    train_data = {}
    for line in train_file.readlines()[1:]:
        line = line.strip().split(',')
        # line[0] 为用户 id，line[1] 为电影 id，line[2] 为评分
        if line[0] not in train_data.keys():
            train_data[line[0]] = {line[1]: line[2]}
        else:
            train_data[line[0]][line[1]] = line[2]
    # * 效应矩阵
    utility_mat = pd.DataFrame(train_data).fillna(0).astype(float)
    return utility_mat

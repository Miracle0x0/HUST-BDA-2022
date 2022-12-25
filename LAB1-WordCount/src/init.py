# -*- coding: utf-8 -*-

"""
# @File       : init.py
# @Time       : 2022/12/22
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 设置文件目录
"""

import os

# ? 文件目录设置
SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
TMP_DIR = os.path.join(BASE_DIR, 'tmp')  # 临时文件目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果文件目录

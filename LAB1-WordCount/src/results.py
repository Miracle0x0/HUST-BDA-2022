# -*- coding: utf-8 -*-

"""
# @File       : results.py
# @Time       : 2022/11/25
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 生成 map-reduce 结果
"""

import os
import threading
from init import *


def generate_subdict(file_name: str, sub_dict: dict) -> None:
    """生成子结果

    Args:
        file_name (str): reduce 结果的文件名
        sub_dict (dict): 子结果存放的字典名称
    """
    f_read = open(os.path.join(TMP_DIR, file_name), 'r')
    for line in f_read:
        line = line.strip()
        word, count = line.split(',', 1)
        sub_dict[word] = count


def generate_result():
    """生成结果，保存在 res 目录下的 result 文件中
    """
    f_write = open(os.path.join(RES_DIR, 'result.csv'), 'w')

    sub_dict1 = {}
    sub_dict2 = {}
    sub_dict3 = {}

    t1 = threading.Thread(target=generate_subdict,
                          args=('reduce1', sub_dict1,))
    t1.start()
    t2 = threading.Thread(target=generate_subdict,
                          args=('reduce2', sub_dict2,))
    t2.start()
    t3 = threading.Thread(target=generate_subdict,
                          args=('reduce3', sub_dict3,))
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    final_dict = {**sub_dict1, **sub_dict2, **sub_dict3}
    fin_list = sorted(final_dict.items(), key=lambda x: x[0])
    for key, value in fin_list:
        f_write.write("{},{}\n".format(key, value))

    print("Result has been generated in 'res' folder.")

# -*- coding: utf-8 -*-

"""
# @File       : apriori.py
# @Time       : 2022/11/27
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: LAB3 A-Priori 算法实现
"""

import os
import time
import numpy as np
import pandas as pd
from threading import Thread

# ? 文件目录设置
SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果文件目录

# ? 常量参数设置
MIN_SUPPORT_OF_FREQ_ITEMS = 0.005  # 频繁项集最小支持度
MIN_CONFIDENCE = 0.5  # 关联规则最小置信度


def get_dataset() -> list[list]:
    """生成数据集

    Returns:
        list[list]: 以二级列表形式存储的数据
    """
    data_set = pd.read_csv(os.path.join(DATA_DIR, 'Groceries.csv'))
    data_items = data_set['items']
    # 处理数据
    array_items = np.array(data_items)
    res_list = []
    for item in array_items:
        item = item.strip('{').strip('}').strip(',')
        goods = []
        for good in item:
            goods.append(good)
        res_list.append(goods)
    # 返回二级列表存储的数据
    return res_list


def item_counter_func(items: list[list], Ck: set):
    item_counter = {}
    for item in items:
        for c in Ck:
            if c.issubset(item):
                if c not in item_counter:
                    item_counter[c] = 1
                else:
                    item_counter[c] += 1
    return item_counter


def filter_apriori(items: list[list], Ck: set, support_items, items_num: int,
                   min_support_of_freq_items: float = MIN_SUPPORT_OF_FREQ_ITEMS) -> set:
    """过滤器（待优化）

    Args:
        items (list[list]): _description_
        Ck (set): _description_
        support_items (_type_): _description_
        items_num (int): _description_
        min_support_of_freq_items (float, optional): _description_. Defaults to MIN_SUPPORT_OF_FREQ_ITEMS.

    Returns:
        set: 由 Ck 生成的 Lk
    """
    Lk = set()
    # print("len(items): ", len(items))
    item_counter = {}
    for item in items:
        for c in Ck:
            if c.issubset(item):
                if c not in item_counter:
                    item_counter[c] = 1
                else:
                    item_counter[c] += 1

    for c in item_counter:
        support_of_items = float(item_counter[c]) / items_num
        if support_of_items >= min_support_of_freq_items:
            Lk.add(c)
            support_items[c] = support_of_items

    return Lk


def create_C1(items: list[list]) -> set:
    """生成 C1

    Args:
        items (list[list]): _description_

    Returns:
        set: _description_
    """
    C1 = set()
    for item in items:
        for good in item:
            if good not in C1:
                good_set = frozenset([good])
                C1.add(good_set)
    return C1


def is_k_sub_apriori(tmp_item, Lk: set) -> bool:
    """判断 k 项集是否是 k + 1 项子集

    Args:
        tmp_item (_type_): _description_
        Lk (set): _description_

    Returns:
        bool: _description_
    """
    for item in tmp_item:
        sub_item = tmp_item - frozenset([item])
        if sub_item not in Lk:
            return False
    return True


def constructor_apriori(Lk: set, k: int, Lk_len: int) -> set:
    Ck1 = set()
    list_Lk = list(Lk)
    list_Lk.sort()

    for i in range(Lk_len):
        for j in range(i + 1, Lk_len):
            # 连接策略：如果 Lk 中某两个元素的前 k - 1 个项相同则可以连接
            item_set1 = list(list_Lk[i])[:k - 1]
            item_set2 = list(list_Lk[j])[:k - 1]
            if item_set1 == item_set2:
                Ck1_tmp_item = list_Lk[i] | list_Lk[j]
                # 剪枝策略：任何非频繁的 k - 1 项集都不是频繁 k 项集的子集
                if is_k_sub_apriori(Ck1_tmp_item, Lk):
                    Ck1.add(Ck1_tmp_item)

    return Ck1


def create_rule(L1: set, L2: set, L3: set,
                support_items_L1: dict, support_items_L2: dict, support_items_L3: dict,
                min_confidence: float = MIN_CONFIDENCE) -> list:
    rule_list = []
    # 寻找 3 阶项集中的关联规则
    for item_k3 in L3:
        for item_k2 in L2:
            if item_k2.issubset(item_k3):
                conf = support_items_L3[item_k3] / support_items_L2[item_k2]
                rule = [item_k2, item_k3 - item_k2, conf]
                if conf >= min_confidence and rule not in rule_list:
                    rule_list.append(rule)
        for item_k1 in L1:
            if item_k1.issubset(item_k3):
                conf = support_items_L3[item_k3] / support_items_L1[item_k1]
                rule = [item_k1, item_k3 - item_k1, conf]
                if conf >= min_confidence and rule not in rule_list:
                    rule_list.append(rule)

    # 寻找 2 阶项集中的关联规则
    for item_k2 in L2:
        for item_k1 in L1:
            if item_k1.issubset(item_k2):
                conf = support_items_L2[item_k2] / support_items_L1[item_k1]
                rule = [item_k1, item_k2 - item_k1, conf]
                if conf >= min_confidence and rule not in rule_list:
                    rule_list.append(rule)

    return rule_list


def save_Lk(filename: str, support_items: dict):
    """保存 Lk

    Args:
        filename (str): 保存文件名
        support_items (dict): _description_
    """
    f_write = open(os.path.join(RES_DIR, filename), 'w')
    f_write.write('{},\t{}\n'.format('frequent-itemSets', 'support'))

    for si in support_items:
        f_write.write('{},\t{}\n'.format(list(si), support_items[si]))
    f_write.write('total: {}'.format(len(support_items)))
    print("{} Done.\n".format(filename))


def save_rule(filename: str, rule_list: list):
    """保存关联规则

    Args:
        filename (str): 保存文件名
        rule_list (list): 关联规则列表
    """
    f_write = open(os.path.join(RES_DIR, filename), 'w')
    f_write.write('----------------rule----------------\n')
    for rule in rule_list:
        f_write.write('{} ==> {}: {}\n'.format(
            list(rule[0]), list(rule[1]), rule[2]
        ))
    f_write.write('total: {}'.format(len(rule_list)))
    print("All done.")


if __name__ == "__main__":
    start_time = time.time()

    # 获取和处理数据
    items = get_dataset()
    # print(items)
    items_len = len(items)

    support_items_L1 = {}
    support_items_L2 = {}
    support_items_L3 = {}

    # 创建 C1 和 L1
    C1 = create_C1(items)
    L1 = filter_apriori(items, C1, support_items_L1, items_len)
    save_Lk('L1', support_items_L1)

    # 创建 C2 和 L2
    C2 = constructor_apriori(L1, 1, len(L1))
    L2 = filter_apriori(items, C2, support_items_L2, items_len)
    save_Lk('L2', support_items_L2)

    # 创建 C3 和 L3
    C3 = constructor_apriori(L2, 2, len(L2))
    L3 = filter_apriori(items, C3, support_items_L3, items_len)
    save_Lk('L3', support_items_L3)

    # 生成关联规则
    rule_list = create_rule(L1, L2, L3, support_items_L1,
                            support_items_L2, support_items_L3)
    save_rule('rule', rule_list)

    finish_time = time.time()
    print("total_time: {} s.".format(finish_time - start_time))

# -*- coding: utf-8 -*-

"""
# @File       : main.py
# @Time       : 2022/12/9
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 
"""

import os
import pandas as pd
import time


# 生成候选频繁一项集
def generate_C1(dataset):
    C1 = set()
    for basket in dataset:
        for item in basket:
            # frozenset:冻结集合，可以作为字典的key
            item = frozenset([item])
            C1.add(item)
    # for item in C1:
    #     print(item)
    return C1


# 根据Ck生成频繁k项集Lk
def generate_Lk(dataset, Ck, min_support: float = 0.005):
    # 字典，记录每个候选频繁项集出现的次数
    itemset_count = {}
    # 频繁k项集
    Lk = set()
    # 记录频繁项集的支持度
    fre_itemset_sup = {}
    for basket in dataset:
        for itemset in Ck:
            if itemset.issubset(basket):
                # 统计候选频繁项集出现次数
                if itemset in itemset_count:
                    itemset_count[itemset] += 1
                else:
                    itemset_count[itemset] = 1
    total = len(dataset)
    for key, value in itemset_count.items():
        # 支持度
        support = value / total
        if support >= min_support:
            Lk.add(key)
            fre_itemset_sup[key] = support
    return Lk, fre_itemset_sup


# 由Lk构造候选频繁项集Ck+1
def generate_next_Ck(Lk, k):
    union_set = set()
    Ck = set()
    for set1 in Lk:
        for set2 in Lk:
            union_set = set1 | set2
            if len(union_set) == k:
                Ck.add(union_set)
    # print(Ck)
    return Ck


# 生成符合置信度要求的关联规则
def generate_rules(L3, sup1, sup2, sup3, min_confidence: float = 0.5):
    rules_list = []
    for fre_itemset in L3:
        union_sup = sup3[fre_itemset]
        for k in range(3):
            tmp = list(fre_itemset)
            set1 = tmp[k]
            tmp.remove(set1)
            set2 = tmp
            conf12 = union_sup / sup1[frozenset([set1])]
            conf21 = union_sup / sup2[frozenset(set2)]
            if conf12 >= min_confidence:
                rules_list.append((set([set1]), set(set2), conf12))
            if conf21 >= min_confidence:
                rules_list.append((set(set2), set([set1]), conf21))
    return rules_list


if __name__ == '__main__':
    dataset = []
    start = time.time()
    data = pd.read_csv("../data/Groceries.csv", header=0)
    for index, row in data.iterrows():
        row_data = row['items'].replace("{", "")
        row_data = row_data.replace("}", "")
        row_data = list(row_data.split(","))
        dataset.append(row_data)
    C1 = generate_C1(dataset)
    L1, fre_itemset_sup1 = generate_Lk(dataset, C1)
    C2 = generate_next_Ck(L1, 2)
    L2, fre_itemset_sup2 = generate_Lk(dataset, C2)
    C3 = generate_next_Ck(L2, 3)
    L3, fre_itemset_sup3 = generate_Lk(dataset, C3)
    rules_list = generate_rules(L3, fre_itemset_sup1, fre_itemset_sup2, fre_itemset_sup3)
    end = time.time()
    print("+++++++++++++++++++++1阶频繁项集+++++++++++++++++++++")
    for key, value in fre_itemset_sup1.items():
        print(set(key), value)
    print("+++++++++++++++++++++2阶频繁项集+++++++++++++++++++++")
    for key, value in fre_itemset_sup2.items():
        print(set(key), value)
    print("+++++++++++++++++++++3阶频繁项集+++++++++++++++++++++")
    for key, value in fre_itemset_sup3.items():
        print(set(key), value)
    print("+++++++++++++++++++++关联规则：+++++++++++++++++++++")
    for rule in rules_list:
        print(rule[0], " => ", rule[1], rule[2])
    print("1阶频繁项集个数为:", len(L1))
    print("2阶频繁项集个数为:", len(L2))
    print("3阶频繁项集个数为:", len(L3))
    print("关联规则个数为:", len(rules_list))
    print("用时:", end - start, 's')

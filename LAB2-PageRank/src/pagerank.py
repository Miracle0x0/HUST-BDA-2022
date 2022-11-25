# -*- coding: utf-8 -*-

"""
# @File       : main.py
# @Time       : 2022/11/25
# @Author     : Asuna
# @Version    : Python 3.9 (Conda)
# @Description: LAB2 PageRank 实现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(__file__)  # 代码目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果目录


def pagerank(mat: np.ndarray, n: int, eps: float = 1e-8):
    r = np.ones(n) / n  # 初始化 R 向量
    err_list = []
    iter_cnt = 0
    while True:
        r_new = np.dot(mat, r)
        r_new /= np.sum(r_new)
        err = np.sqrt(np.sum(np.square(r_new - r)))
        err_list.append(err)
        if err < eps:
            break
        r = r_new.copy()
        iter_cnt += 1
    return r_new, err_list, iter_cnt


def pagerank_teleport(mat: np.ndarray, n: int, eps: float = 1e-8, beta: float = 0.85):
    r = np.ones(n) / n  # 初始化 R 向量
    err_list = []
    iter_cnt = 0
    A = beta * mat + (1 - beta) * np.ones((n, n)) / n
    while True:
        r_new = np.dot(A, r)
        r_new /= np.sum(r_new)
        err = np.sqrt(np.sum(np.square(r_new - r)))
        err_list.append(err)
        if err < eps:
            break
        r = r_new.copy()
        iter_cnt += 1
    return r_new, err_list, iter_cnt


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'sent_receive.csv'))
    persons = pd.read_csv(os.path.join(DATA_DIR, 'Persons.csv'))
    N = persons.shape[0]
    # * 创建矩阵 M
    M = np.zeros((N, N))

    for (sent_id, tmp_df) in list(df.groupby('sent_id')):
        receivers = []
        for (receiver_id, _) in list(tmp_df.groupby('receive_id')):
            receivers.append(int(receiver_id))
        # * 矩阵归一化
        M[receivers, int(sent_id)] = 1 / len(receivers)

    print("M:\n", M)

    # * 无修正因子算法
    print("无修正因子算法")
    r1, err1, iter_cnt1 = pagerank(M, N)
    print("r1:\n", r1)
    print("迭代次数: ", iter_cnt1)

    # * 含修正因子算法
    print("含修正因子算法")
    r2, err2, iter_cnt2 = pagerank_teleport(M, N)
    print("r2:\n", r2)
    print("迭代次数: ", iter_cnt2)

    # * 保存结果
    with open(os.path.join(RES_DIR, 'result'), 'w') as f_write:
        idx = 0
        while idx < len(r2):
            f_write.write(str(idx + 1) + "\t" + str(float(r2[idx])) + "\n")
            idx += 1

    persons['pagerank'] = r1
    persons['pagerank_teleport'] = r2
    persons.to_csv(os.path.join(RES_DIR, 'rank_value.csv'))

    print("All done.")

    # * 分析误差
    fig = plt.figure()
    plt.plot(err1, label="PageRank")
    plt.plot(err2, label="PageRank with teleport beta=0.85")
    plt.legend()
    plt.title("Err - Iterations")
    plt.xlabel("iter")
    plt.ylabel("error")
    plt.show()

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

# ? 文件目录设置
SRC_DIR = os.path.dirname(__file__)  # 代码目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果目录


def get_filename(file_name: str) -> str:
    """获取文件

    Args:
        file_name (str): 文件名

    Returns:
        (str): 包含路径的完整文件名
    """
    return os.path.join(DATA_DIR, file_name)


def pagerank(mat: np.ndarray, n: int, eps: float = 1e-8):
    """pagerank 算法实现

    Args:
        mat (np.ndarray): 邻接矩阵
        n (int): 结点数量
        eps (float, optional): 迭代误差. 默认值 1e-8.

    Returns:
        r_rew, err_list, iter_cnt, err: 秩向量，错误率，迭代次数，退出时误差
    """
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
    return r_new, err_list, iter_cnt, err


def pagerank_teleport(mat: np.ndarray, n: int, eps: float = 1e-8, beta: float = 0.85):
    """pagerank 算法（含阻尼系数）实现

    Args:
        mat (np.ndarray): 邻接矩阵
        n (int): 结点数量
        eps (float, optional): 迭代误差. 默认值 1e-8.
        beta (float, optional): 阻尼系数. 默认值 0.85.

    Returns:
        r_rew, err_list, iter_cnt, err: 秩向量，错误率，迭代次数，退出时误差
    """
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
    return r_new, err_list, iter_cnt, err


def save_result(r: np.ndarray) -> None:
    """保存结果

    Args:
        r (np.ndarray): 秩向量
    """
    with open(os.path.join(RES_DIR, 'result'), 'w') as f_write:
        idx = 0
        while idx < len(r):
            f_write.write(str(idx + 1) + "\t" + str(float(r[idx])) + "\n")
            idx += 1


if __name__ == "__main__":
    df = pd.read_csv(get_filename('sent_receive.csv'))
    persons = pd.read_csv(get_filename('Persons.csv'))
    N = persons.shape[0]
    # * 创建矩阵 M
    M = np.zeros((N, N))

    for (sent_id, tmp_df) in list(df.groupby('sent_id')):
        receivers = []
        for (receiver_id, _) in list(tmp_df.groupby('receive_id')):
            receivers.append(int(receiver_id))
        M[receivers, int(sent_id)] = 1
    # * 矩阵归一化
    col_sum = np.sum(M, axis=0)
    for col in range(M.shape[1]):
        if col_sum[col] != 0:
            M[:, col] /= col_sum[col]

    print("归一化后的邻接矩阵 M:\n", M)

    # * 无修正因子算法
    print("无修正因子算法")
    r1, err1, iter_cnt1, last_err = pagerank(M, N)
    print("r1:\n", r1)
    print("迭代次数: ", iter_cnt1)
    print("最后一次误差: ", last_err)
    print("r1 中因子之和: ", np.sum(r1).squeeze())

    # * 含修正因子算法
    print("含修正因子算法")
    r2, err2, iter_cnt2, last_err = pagerank_teleport(M, N)
    print("r2:\n", r2)
    print("迭代次数: ", iter_cnt2)
    print("最后一次误差: ", last_err)
    print("r2 中因子之和: ", np.sum(r2).squeeze())

    # * 保存结果
    save_result(r2)

    persons['pagerank'] = r1
    persons['pagerank_teleport'] = r2
    persons.to_csv(get_filename('rank_value.csv'))

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

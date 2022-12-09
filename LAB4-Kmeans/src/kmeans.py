# -*- coding: utf-8 -*-

"""
# @File       : kmeans.py
# @Time       : 2022/11/28
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: LAB4 Kmeans 算法实现
"""

import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ? 画图字体设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

# ? 文件目录设置
SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
RES_DIR = os.path.join(BASE_DIR, 'res')  # 结果文件目录


def get_data(filename: str) -> tuple[np.ndarray, int]:
    """获取数据

    Args:
        filename (str): 数据文件名

    Returns:
        tuple[np.ndarray, int]: 数据文件和数据长度
    """
    samples = []
    with open(os.path.join(DATA_DIR, filename)) as f_read:
        tmp_data = csv.reader(f_read)
        for row in tmp_data:
            samples.append(row)
    samples = np.array([[float(x) for x in row] for row in samples])
    sample_num = len(samples)
    return samples, sample_num


def get_distance(center: np.ndarray, sample: np.ndarray) -> float:
    """计算向量之间的欧几里得距离

    Args:
        center (np.ndarray): vector1
        sample (np.ndarray): vector2

    Returns:
        float: euclidean distance of vector1 and vector2
    """
    return np.linalg.norm(center - sample)


def get_sse(process_mindata: np.mat, sample_num: int, k: int = 3) -> float:
    """计算距离平方和

    Args:
        process_mindata (np.mat): 数值矩阵
        sample_num (int): 样本数量
        k (int, optional): 聚类中心数量. Defaults to 3.

    Returns:
        float: 距离平方和
    """
    sse_num = np.zeros(3)
    sse = 0
    for i in range(sample_num):
        sse_num[int(process_mindata[i, 1]) - 1] += process_mindata[i, 0]
    for i in range(k):
        print("The ", i + 1, sse_num[i])
    sse += sum(sse_num)
    print("All sse: ", sse)
    return sse


def get_acc(process_mindata: np.mat, sample_num: int, k: int = 3) -> float:
    """计算准确率

    Args:
        process_mindata (np.mat): 数值矩阵
        sample_num (int): 样本数量
        k (int, optional): 聚类中心数量. Defaults to 3.

    Returns:
        float: 准确率
    """
    clusters = np.zeros((k, k))
    # 类 1: 59
    # 类 2: 71
    # 类 3: 48
    for i in range(sample_num):
        idx2 = int(process_mindata[i, 1]) - 1
        idx1 = 0 if i < 60 else 1 if i < 130 else 2
        clusters[idx1][idx2] += 1
    acc = np.sum(np.max(clusters, axis=0))
    acc /= sample_num
    print("acc: ", acc)
    return acc


def rand_center(data: np.ndarray, k: int = 3) -> np.ndarray:
    """生成随机聚类中心

    Args:
        data (np.ndarray): 数据集
        k (int, optional): 聚类中心数量. Defaults to 3.

    Returns:
        np.ndarray: 随机聚类中心
    """
    n = data.shape[1]
    centdroids = np.zeros((k, n))
    for j in range(n):
        minJ = min(data[:, j])
        maxJ = max(data[:, j])
        rangeJ = float(maxJ - minJ)
        centdroids[:, j] = minJ + rangeJ * np.random.rand(k)

    return centdroids


if __name__ == "__main__":
    # * Part 1 计算部分
    # 提取数据
    samples, sample_num = get_data('normalizedwinedata.csv')
    # 设定 k，初始化 k 个聚类中心，13 维度
    k = 3
    # cluster_centers = np.random.random((k, len(samples[0])))
    cluster_centers = rand_center(samples)
    # 矩阵行指示样点，第 1 列指示过程最小距离，第 2 列指示所属类簇
    process_mindata = np.mat(np.zeros((sample_num, 2)))
    # process_mindata[:, 1] = 0
    # 设置最大迭代次数为 100
    updated = True
    iter_counter = 0
    while updated and iter_counter < 100:
        updated = False
        for i in range(sample_num):
            min_distance = float('inf')
            min_center = 0
            for j in range(k):
                distance = get_distance(cluster_centers[j], samples[i])
                if distance < min_distance:
                    min_distance = distance
                    min_center = j + 1
            if process_mindata[i, 0] != min_distance or (process_mindata[i, 1] != min_center and min_center != 0):
                process_mindata[i, 0] = min_distance
                process_mindata[i, 1] = min_center
                updated = True
        # 更新类簇中心
        for i in range(k):
            # 获取当前类簇的样本点
            new_point = []
            for j in range(sample_num):
                if process_mindata[j, 1] == i + 1:
                    new_point.append(samples[j])
            # if new_point == []:
            #     print("Empty list!")
            #     continue
            cluster_centers[i, :] = np.mean(new_point, axis=0)

        iter_counter += 1

    print("iteration times: ", iter_counter)
    sse = get_sse(process_mindata, sample_num)
    acc = get_acc(process_mindata, sample_num)

    # * Part 2 作图部分
    X = 6  # 总酚
    Y = 7  # 黄酮

    plt.xlabel('总酚')
    plt.ylabel('黄酮')
    plt.title('SSE=%.3f Acc=%.3f' % (sse, acc))
    plt.axis([0, 1, 0, 1])
    for i in range(sample_num):
        if int(process_mindata[i, 1]) == 1:
            plt.scatter(samples[i][X], samples[i][Y], c='r')
        elif int(process_mindata[i, 1]) == 2:
            plt.scatter(samples[i][X], samples[i][Y], c='g')
        else:
            plt.scatter(samples[i][X], samples[i][Y], c='b')
    plt.show()
    print("All done.")

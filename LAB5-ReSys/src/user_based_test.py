# -*- coding: utf-8 -*-

"""
# @File       : user_based.py
# @Time       : 2022/12/9
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: User-User 的协同过滤算法 k 值选择分析
"""

import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from init import *
from user_based import UserBasedRecommendSystem, MinHashUserRecommendSystem

# ? 画图字体设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':
    utility_matrix = get_utility_mat()

    # ? 测试数据读取
    test_data = pd.read_csv(get_filename('test_set.csv'))
    test_data.drop('timestamp', axis=1, inplace=True)
    test_users, test_movies, test_ratings = test_data[
        'userId'], test_data['movieId'], test_data['rating'].values

    # * 基础版
    base_recommender = UserBasedRecommendSystem(utility_matrix)
    base_recommender.get_corr_mat()

    # ? 使用不同的 k 值进行预测
    k_array = np.arange(50, 310, 10)  # k 值列表
    sse_array = np.zeros(len(k_array))
    for i in range(len(k_array)):
        k = k_array[i]
        # 进行预测
        pred_ratings = np.zeros(len(test_data))
        for j in range(len(test_data)):
            pred_ratings[j] = base_recommender.predict(test_users[j], test_movies[j], k)
        # 计算 SSE
        sse_array[i] = np.sum(np.square(pred_ratings - test_ratings))
    print("sse_array:")
    print(sse_array)

    # ? 绘制 SSE 随 k 变化趋势图
    x = k_array
    res = sse_array
    x_new = np.linspace(np.min(x), np.max(x), 3000)
    func = interp1d(x, res, kind='cubic')
    y_new = func(x_new)
    plt.plot(x_new, y_new)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [math.floor(np.min(res)), res[i]], c='r', linestyle='--')
    plt.title('SSE 随 k 变化趋势图')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.ylim(math.floor(np.min(res)))
    plt.show()

    # # * 进阶版
    # minhash_recommender = MinHashUserRecommendSystem(utility_matrix)
    # minhash_recommender.get_signature_mat()
    # minhash_recommender.get_minhash_sim_mat()
    #
    # # ? 使用不同的 k 值进行预测
    # k_array = np.arange(50, 310, 10)  # k 值列表
    # sse_array = np.zeros(len(k_array))
    # for i in range(len(k_array)):
    #     k = k_array[i]
    #     # 进行预测
    #     pred_ratings = np.zeros(len(test_data))
    #     for j in range(len(test_data)):
    #         pred_ratings[j] = minhash_recommender.predict(test_users[j], test_movies[j], k)
    #     # 计算 SSE
    #     sse_array[i] = np.sum(np.square(pred_ratings - test_ratings))
    # print("sse_array:")
    # print(sse_array)
    #
    # # ? 绘制 SSE 随 k 变化趋势图
    # x = k_array
    # res = sse_array
    # x_new = np.linspace(np.min(x), np.max(x), 3000)
    # func = interp1d(x, res, kind='cubic')
    # y_new = func(x_new)
    # plt.plot(x_new, y_new)
    # for i in range(len(x)):
    #     plt.plot([x[i], x[i]], [math.floor(np.min(res)), res[i]], c='r', linestyle='--')
    # plt.title('SSE 随 k 变化趋势图')
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.ylim(math.floor(np.min(res)))
    # plt.show()

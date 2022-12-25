# -*- coding: utf-8 -*-

"""
# @File       : content_based.py
# @Time       : 2022/12/9
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 基于内容的推荐算法 k 值选择分析
"""

import sys
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.interpolate import interp1d

from init import *
from content_based import ContentBasedRecommendSystem, MinHashContentRecommendSystem

# ? 画图字体设置
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置横坐标为整数


def process_bar(num, total):
    """进度条打印"""
    rate = float(num) / total
    rate_num = int(100 * rate)
    r = '\r[{}{}]{}%'.format('*' * rate_num, ' ' * (100 - rate_num), rate_num)
    sys.stdout.write(r)
    sys.stdout.flush()
    if rate == 1:
        print()


if __name__ == '__main__':
    utility_matrix = get_utility_mat()

    # ? 电影数据读取
    movies = pd.read_csv(get_filename('movies.csv'))

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies['genres'].tolist()).toarray()

    # ? 测试数据读取
    test_data = pd.read_csv(get_filename('test_set.csv'))
    test_data.drop('timestamp', axis=1, inplace=True)
    test_users, test_movies, test_ratings = test_data[
        'userId'], test_data['movieId'], test_data['rating'].values

    # * 进阶版
    minhash_recommender = MinHashContentRecommendSystem(utility_matrix, movies, tfidf_matrix)
    minhash_recommender.get_movie_sim_mat()

    # ? 使用不同的哈希函数数量 n 进行预测
    n_array = np.arange(10, 32, 2)  # 哈希函数数量 n 列表
    sse_array = np.zeros(len(n_array))
    for i in range(len(n_array)):
        # print("I: ", i)
        process_bar(i + 1, len(n_array))
        minhash_recommender.get_signature_mat(n_array[i])
        minhash_recommender.get_minhash_sim_mat()
        # 进行预测
        pred_ratings = np.zeros(len(test_data))
        for j in range(len(test_data)):
            pred_ratings[j] = minhash_recommender.predict(test_users[j], test_movies[j])
        # 计算 SSE
        sse_array[i] = np.sum(np.square(pred_ratings - test_ratings))
    print("sse_array:")
    print(sse_array)

    # ? 绘制 SSE 随哈希函数数量 n 变化趋势图
    x = n_array
    res = sse_array
    x_new = np.linspace(np.min(x), np.max(x), 3000)
    func = interp1d(x, res, kind='cubic')
    y_new = func(x_new)
    plt.plot(x_new, y_new)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [math.floor(np.min(res)), res[i]], c='r', linestyle='--')
    plt.title('SSE 随哈希函数数量 n 变化趋势图')
    plt.xlabel('哈希函数数量 n')
    plt.ylabel('SSE')
    plt.ylim(math.floor(np.min(res)))
    plt.show()

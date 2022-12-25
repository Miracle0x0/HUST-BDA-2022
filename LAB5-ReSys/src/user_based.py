# -*- coding: utf-8 -*-

"""
# @File       : user_based.py
# @Time       : 2022/12/9
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: User-User 的协同过滤算法
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from init import *


class UserBasedRecommendSystem(object):
    """基于用户的协同过滤推荐系统"""

    def __init__(self, utility_mat):
        self.utility_mat = utility_mat
        self.user_sim_mat = None

    def get_corr_mat(self):
        """计算 Pearson 相关系数矩阵"""
        self.user_sim_mat = self.utility_mat.corr()

    # TODO k 可以选择
    def predict(self, user_id: int, movie_id: int, k: int = 110) -> float:
        """根据 k 个最相似的用户预测用户 user_id 对电影 movie_id 的评分

        Args:
            user_id (int): 用户 id
            movie_id (int): 电影 id
            k (int, optional): 选取的相似用户数量. 默认为 110.

        Returns:
            float: 用户的电影的预测评分
        """
        user_id = str(user_id)
        movie_id = str(movie_id)
        # 从相关系数矩阵中找到与 user_id 相关的用户
        sim_dict = dict(self.user_sim_mat[user_id])
        sorted_sim_dict = sorted(
            sim_dict.items(), key=lambda x: x[1], reverse=True)
        # 取 k 个最相似的用户的 id
        top_k_id = [sorted_sim_dict[i][0] for i in range(k)]
        top_k_mat = self.utility_mat[top_k_id]
        # 获得 k 个最相似用户对 movie_id 的评分
        scores = top_k_mat.loc[movie_id]
        pred_score = np.mean(scores[scores != 0])
        return pred_score

    def recommend(self, user_id: str, k: int, n: int) -> None:
        """根据 k 个最相似的用户为用户 user_id 推荐 n 部电影

        Args:
            user_id (str): 用户 id
            k (int): 选取的相似用户数量
            n (int): 推荐的电影数量
        """
        # 从相关系数矩阵中找到与 user_id 相关的用户
        sim_dict = dict(self.user_sim_mat[user_id])
        sorted_sim_dict = sorted(
            sim_dict.items(), key=lambda x: x[1], reverse=True)
        # 取 k 个最相似的用户的 id
        top_k_id = [sorted_sim_dict[i][0] for i in range(k)]
        top_k_mat = self.utility_mat[top_k_id]
        pred_dict = {}
        for i in range(len(self.utility_mat)):
            x = top_k_mat.iloc[i]
            if len(x[x != 0]) > 20:  # * 某部电影至少有 20 个相关用户打过分才进行预测
                pred_i = np.mean(x[x != 0])
                pred_dict[i] = 0 if np.isnan(pred_i) else pred_i
            else:
                pred_dict[i] = 0
        # 对预测的电影按照预测分数进行降序排列
        sorted_pred_dict = sorted(
            pred_dict.items(), key=lambda d: d[1], reverse=True)
        # 取前 n 个电影进行推荐
        pred_res = sorted_pred_dict[:min(n, len(sorted_pred_dict))]
        # 推荐结果
        print("对用户 {} 推荐如下电影:".format(user_id))
        print("Movie\tScore")
        print("-" * 15)
        for i in range(n):
            idx, score = pred_res[i]
            print("%-6s\t%.3f" % (str(self.utility_mat.index[idx]), score))


class MinHashUserRecommendSystem(UserBasedRecommendSystem):
    """引入 MinHash 算法的基于用户的协同过滤推荐系统"""

    def __init__(self, utility_mat):
        super().__init__(utility_mat)
        self.array_utility_mat = np.array(self.utility_mat)
        self.signature_mat = None
        self.minhash_sim_mat = None

    def get_signature_mat(self, hash_func_cnt: int = 10) -> None:
        """生成签名矩阵

        Args:
            hash_func_cnt (int, optional): 哈希函数个数. 默认为 10.
        """
        um = np.where(self.array_utility_mat <= 2.5, 0, 1)
        self.signature_mat = np.zeros(
            (hash_func_cnt, self.array_utility_mat.shape[1]))
        # * 随机数种子列表，用于固定哈希函数
        seed_array = np.arange(hash_func_cnt)
        for i in range(hash_func_cnt):
            np.random.seed(seed_array[i])
            np.random.shuffle(um)
            self.signature_mat[i] = np.array(
                [np.where(um[:, j] == 1)[0][0] for j in range(um.shape[1])])

    def cal_minhash_sim(self, idx: int) -> np.ndarray:
        """计算 MinHash 相似度

        Args:
            idx (int): MinHash 矩阵行序号

        Returns:
            np.ndarray: MinHash 相似矩阵的 idx 行
        """
        tmp_mat = self.signature_mat - \
                  self.signature_mat[:, idx].reshape(-1, 1)
        res_mat = 1 - np.count_nonzero(tmp_mat, axis=0) / tmp_mat.shape[0]
        res_mat[res_mat == 1] = 0
        return res_mat

    def get_minhash_sim_mat(self) -> None:
        """计算 MinHash 相似矩阵"""
        user_num = self.signature_mat.shape[1]
        self.minhash_sim_mat = np.zeros((user_num, user_num))
        for i in range(user_num):
            self.minhash_sim_mat[i] = self.cal_minhash_sim(i)

        self.user_sim_mat = pd.DataFrame(self.minhash_sim_mat, index=range(1, self.minhash_sim_mat.shape[1] + 1),
                                         columns=range(1, self.minhash_sim_mat.shape[1] + 1)).astype(float)

    # TODO k 可以选择
    def predict(self, user_id: int, movie_id: int, k: int = 110) -> float:
        """根据 k 个最相似的用户预测用户 user_id 对电影 movie_id 的评分

        Args:
            user_id (int): 用户 id
            movie_id (int): 电影 id
            k (int, optional): 选取的相似用户数量. 默认为 110.

        Returns:
            float: 用户的电影的预测评分
        """
        # 从相关系数矩阵中找到与 user_id 相关的用户
        movie_id = str(movie_id)
        sim_dict = dict(self.user_sim_mat[user_id])
        sorted_sim_dict = sorted(
            sim_dict.items(), key=lambda x: x[1], reverse=True)
        # 取 k 个最相似的用户的 id
        top_k_id = [str(sorted_sim_dict[i][0]) for i in range(k)]
        top_k_mat = self.utility_mat[top_k_id]
        # 获得 k 个最相似用户对 movie_id 的评分
        scores = top_k_mat.loc[movie_id]
        pred_score = np.mean(scores[scores != 0])
        return pred_score

    def recommend(self, user_id: str, k: int, n: int) -> None:
        """根据 k 个最相似的用户对用户 user_id 推荐 n 部电影

        Args:
            user_id (str): 用户 id
            k (int): 选取的相似用户数量
            n (int): 推荐的电影数量
        """
        user_id = int(user_id)
        # 从相关系数矩阵中找到与 user_id 相关的用户
        sim_dict = dict(self.user_sim_mat[user_id])
        sorted_sim_dict = sorted(
            sim_dict.items(), key=lambda x: x[1], reverse=True)
        # 取 k 个最相似的用户的 id
        top_k_id = [str(sorted_sim_dict[i][0]) for i in range(k)]
        top_k_mat = self.utility_mat[top_k_id]
        pred_dict = {}
        for i in range(len(self.utility_mat)):
            x = top_k_mat.iloc[i]
            if len(x[x != 0]) > 20:  # * 某部电影至少有 20 个相关用户打过分才进行预测
                pred_i = np.mean(x[x != 0])
                pred_dict[i] = 0 if np.isnan(pred_i) else pred_i
            else:
                pred_dict[i] = 0
        # 对预测的电影按照预测分数进行降序排列
        sorted_pred_dict = sorted(
            pred_dict.items(), key=lambda d: d[1], reverse=True)
        # 取前 n 个电影进行推荐
        pred_res = sorted_pred_dict[:min(n, len(sorted_pred_dict))]
        # 推荐结果
        print("对用户 {} 推荐如下电影:".format(user_id))
        print("Movie\tScore")
        print("-" * 15)
        for i in range(n):
            idx, score = pred_res[i]
            print("%-6s\t%.3f" % (str(self.utility_mat.index[idx]), score))


if __name__ == "__main__":
    utility_matrix = get_utility_mat()

    # ? 测试数据读取
    test_data = pd.read_csv(get_filename('test_set.csv'))
    test_data.drop('timestamp', axis=1, inplace=True)
    test_users, test_movies, test_ratings = test_data[
        'userId'], test_data['movieId'], test_data['rating'].values

    # * 基础版
    start_time = time.time()

    base_recommender = UserBasedRecommendSystem(utility_matrix)
    base_recommender.get_corr_mat()

    pred_ratings = np.zeros(len(test_data))
    # * 进行预测
    for i in range(len(test_data)):
        pred_ratings[i] = base_recommender.predict(
            test_users[i], test_movies[i])

    # * 计算 SSE
    sse = np.sum(np.square(pred_ratings - test_ratings))
    print("基础版 SSE = ", sse)

    finish_time = time.time()
    # * 用时统计
    print("总时间: {:.3f} s.".format(finish_time - start_time))

    # * 为指定用户进行推荐
    base_recommender.recommend('300', 100, 10)

    print("\n----------------------\n")

    # * MinHash 版
    start_time = time.time()

    minhash_recommender = MinHashUserRecommendSystem(utility_matrix)
    minhash_recommender.get_signature_mat()
    minhash_recommender.get_minhash_sim_mat()

    # * 进行预测
    for i in range(len(test_data)):
        pred_ratings[i] = minhash_recommender.predict(
            test_users[i], test_movies[i])

    # * 计算 SSE
    sse = np.sum(np.square(pred_ratings - test_ratings))
    print("MinHash 版 SSE = ", sse)

    finish_time = time.time()
    # * 用时统计
    print("总时间: {:.3f} s.".format(finish_time - start_time))

    # * 为指定用户进行推荐
    minhash_recommender.recommend('300', 100, 10)

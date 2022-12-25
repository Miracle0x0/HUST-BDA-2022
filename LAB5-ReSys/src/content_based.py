# -*- coding: utf-8 -*-

"""
# @File       : content_based.py
# @Time       : 2022/12/9
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 基于内容的推荐算法
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from init import *

# ? 电影数据读取
movies = pd.read_csv(get_filename('movies.csv'))

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'].tolist()).toarray()


class ContentBasedRecommendSystem(object):
    """基于内容的推荐系统"""

    def __init__(self, utility_mat, movies, tfidf_matrix):
        self.utility_mat = utility_mat
        self.movies = movies
        # 索引 - movieId 映射
        self.index_to_id = dict(enumerate(self.movies['movieId']))
        # movieId - 索引映射
        self.id_to_index = dict(
            zip(self.index_to_id.values(), self.index_to_id.keys()))

        self.tfidf_mat = tfidf_matrix  # tf-idf 特征矩阵
        self.movie_sim_mat = None  # 电影相似度矩阵

    def get_movie_sim_mat(self):
        """利用余弦相似度计算电影之间的相似度矩阵"""
        self.movie_sim_mat = cosine_similarity(self.tfidf_mat)

    def get_predict_score(self, rated_score: np.ndarray, rated_id: np.ndarray, movie_id: int) -> float:
        """计算预测值

        Args:
            rated_score (np.ndarray): _description_
            rated_id (np.ndarray): _description_
            movie_id (int): 电影 id

        Returns:
            float: 预测值
        """
        distances = self.movie_sim_mat[movie_id]  # movie_id 与其它电影的相似度
        # 计算集合
        computed_dict = {}
        for i in range(len(rated_id)):
            cosine = distances[self.id_to_index[rated_id[i]]]
            if cosine > 0:
                computed_dict[i] = cosine
        if len(computed_dict.keys()):  # 计算集合不为空，则计算加权预测值
            score_sum, sim_sum = 0, 0
            for k, v in computed_dict.items():
                score_sum += rated_score[k] * v
                sim_sum += v
            return score_sum / sim_sum
        else:  # 计算集合为空，则计算平均值
            return np.mean(rated_score)

    def predict(self, user_id: int, movie_id: int) -> float:
        """预测用户 user_id 对电影 movie_id 的打分

        Args:
            user_id (int): 用户 id
            movie_id (int): 电影 id

        Returns:
            float: 预测评分
        """
        user_id = str(user_id)
        # 选取用户打过分的电影
        exist_rating = (self.utility_mat[user_id] != 0)
        rated = self.utility_mat[user_id][exist_rating]
        # 打过分的所有电影的分值
        rated_score = np.array(rated.array)
        # 打过分的所有电影的 id
        rated_id = np.array(rated.index).astype(int)
        return self.get_predict_score(rated_score, rated_id, self.id_to_index[movie_id])

    def recommend(self, user_id: int, k: int) -> None:
        """为用户 user_id 推荐 k 部电影

        Args:
            user_id (int): 用户 id
            k (int): 推荐电影数量
        """
        user_id = str(user_id)
        # 选取用户打过分的电影
        exist_rating = (self.utility_mat[user_id] != 0)
        rated = self.utility_mat[user_id][exist_rating]
        # 打过分的所有电影的分值
        rated_score = np.array(rated.array)
        # 打过分的所有电影的 id
        rated_id = np.array(rated.index).astype(int)
        rec_movies = {}
        for i in range(len(self.movies)):
            idx = self.movies['movieId'][i]
            title = self.movies['title'][i]
            if idx not in rated_id:
                rec_movies[(idx, title)] = self.get_predict_score(
                    rated_score, rated_id, self.id_to_index[idx])
        # 排序
        rec_movies_items = list(rec_movies.items())
        rec_movies_items.sort(key=lambda x: x[1], reverse=True)
        rec_movies = [(key, value) for key, value in rec_movies_items][:k]
        # 推荐结果
        print("对用户 {} 推荐如下电影:".format(user_id))
        print("%-10s\t%-40s\t%-10s" % ("Movie", "Title", "Score"))
        print("-" * 61)
        for item in rec_movies:
            print("%-10s\t%-40s\t%.3f" % (item[0][0], item[0][1], item[1]))


class MinHashContentRecommendSystem(ContentBasedRecommendSystem):
    """引入 MinHash 算法的基于内容的推荐系统"""

    def __init__(self, utility_mat, movies, tfidf_matrix):
        super().__init__(utility_mat, movies, tfidf_matrix)
        self.signature_mat = None
        self.minhash_sim_mat = None

    def get_signature_mat(self, hash_func_cnt: int = 20) -> None:
        """生成签名矩阵

        Args:
            hash_func_cnt (int, optional): 哈希函数个数. 默认为 20.
        """
        tm = np.array(self.tfidf_mat).transpose()
        tm[tm != 0] = 1
        self.signature_mat = np.zeros((hash_func_cnt, tm.shape[1]))
        # * 随机数种子列表，用户固定随机数种子
        seed_array = np.arange(hash_func_cnt)
        for i in range(hash_func_cnt):
            np.random.seed(seed_array[i])
            np.random.shuffle(tm)
            self.signature_mat[i] = np.array(
                [np.where(tm[:, j] == 1)[0][0] for j in range(tm.shape[1])])

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

    def get_minhash_sim_mat(self):
        """计算 MinHash 相似矩阵"""
        movie_num = self.signature_mat.shape[1]
        self.minhash_sim_mat = np.zeros((movie_num, movie_num))
        for i in range(movie_num):
            self.minhash_sim_mat[i] = self.cal_minhash_sim(i)

    def get_predict_score(self, rated_score: np.ndarray, rated_id: np.ndarray, movie_id: int):
        """计算预测值"""
        distances = self.minhash_sim_mat[movie_id]  # movie_id 与其它电影的相似度
        # 计算集合
        computed_dict = {}
        for i in range(len(rated_id)):
            cosine = distances[self.id_to_index[rated_id[i]]]
            if cosine > 0:
                computed_dict[i] = cosine
        if len(computed_dict.keys()):  # 计算集合不为空，则计算加权预测值
            score_sum, sim_sum = 0, 0
            for k, v in computed_dict.items():
                score_sum += rated_score[k] * v
                sim_sum += v
            return score_sum / sim_sum
        else:  # 计算集合为空，则计算平均值
            return np.mean(rated_score)


if __name__ == "__main__":
    utility_matrix = get_utility_mat()

    # ? 测试数据读取
    test_data = pd.read_csv(get_filename('test_set.csv'))
    test_data.drop('timestamp', axis=1, inplace=True)
    test_users, test_movies, test_ratings = test_data[
        'userId'], test_data['movieId'], test_data['rating'].values

    # * 基础版
    start_time = time.time()

    base_recommender = ContentBasedRecommendSystem(
        utility_matrix, movies, tfidf_matrix)
    base_recommender.get_movie_sim_mat()

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
    base_recommender.recommend(83, 10)

    print("\n----------------------\n")

    # * MinHash 版
    start_time = time.time()

    minhash_recommender = MinHashContentRecommendSystem(
        utility_matrix, movies, tfidf_matrix)
    minhash_recommender.get_movie_sim_mat()
    minhash_recommender.get_signature_mat()
    minhash_recommender.get_minhash_sim_mat()

    # * 进行预测
    for i in range(len(test_data)):
        pred_ratings[i] = minhash_recommender.predict(
            test_users[i], test_movies[i])

    # * 计算 SSE
    sse = np.sum(np.square(pred_ratings - test_ratings))
    print("Minhash 版 SSE = ", sse)

    finish_time = time.time()
    # * 用时统计
    print("总时间: {:.3f} s.".format(finish_time - start_time))

    # * 为指定用户进行推荐
    minhash_recommender.recommend(83, 10)

# -*- coding: utf-8 -*-

"""
# @File       : reducer.py
# @Time       : 2022/11/24
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 实现 reduce 功能
"""

import os
import re
import threading
import time
from init import *

REDUCE_NODE_COUNT = 3  # reduce 节点个数

# 线程池 (reduce)
reduce_thread_pool = []
# 开始时间
start_time = 0


def create_thread(idx: int) -> threading.Thread:
    """创建线程 (reducer)

    Args:
        idx (int): 待创建的线程序号

    Returns:
        threading.Thread: 新创建的线程
    """
    src_dir = os.path.join(TMP_DIR, 'shuffle' + str(idx))  # 源文件目录
    dst_dir = os.path.join(TMP_DIR, 'reduce' + str(idx))  # reduce 文件输出目录
    return threading.Thread(target=reducer(src_dir, dst_dir), args=('t' + str(idx),), name='t' + str(idx))


def join_thread(idx: int, timer_on: bool = True) -> None:
    """阻塞线程 (reducer)

    Args:
        idx (int): 待阻塞的线程序号
        timer_on (bool): 是否启用计时，默认启用
    """
    reduce_thread_pool[idx - 1].join()
    if timer_on:
        print("t%d cost %s s." % (idx, time.perf_counter() - start_time))


def reducer(src_file: str, dst_file: str):
    """reduce 功能函数

    Args:
        src_file (str): 源文件路径
        dst_file (str): 目标文件路径
    """
    f_read = open(src_file, 'r')
    f_write = open(dst_file, 'w')
    count_dict = {}

    for line in f_read:
        line = line.strip()
        word, count = line.split(',', 1)
        try:
            count = int(count)
        except ValueError:
            continue
        if word in count_dict.keys():
            count_dict[word] += count
        else:
            count_dict[word] = count

    count_dict = sorted(
        count_dict.items(), key=lambda x: x[0], reverse=False)

    for k, v in count_dict:
        f_write.write("{},{}\n".format(k, v))


class Reducer(threading.Thread):
    def __init__(self, reduce_node_count: int):
        super().__init__()
        self.reduce_node_count = reduce_node_count
        self.cur_node_count = 0
        self.reduce_thread_list = []
        self.lock = threading.Lock()

    def create_reduce(self, idx: int):
        # print("reduce thread %d start!" % idx)
        reduce_thread = create_thread(idx)
        reduce_thread.start()
        reduce_thread.join()
        # print("reduce thread %d finish!" % idx)

    def run(self):
        while self.cur_node_count < self.reduce_node_count:
            idx = self.cur_node_count + 1
            reduce_t = threading.Thread(
                target=self.create_reduce, args=(idx,))
            self.reduce_thread_list.append(reduce_t)
            reduce_t.start()
            self.lock.acquire()
            self.cur_node_count += 1
            self.lock.release()

        for i in range(self.reduce_node_count):
            self.reduce_thread_list[i].join()
        print("reduce threads all finish.")

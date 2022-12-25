# -*- coding: utf-8 -*-

"""
# @File       : shuffler.py
# @Time       : 2022/11/24
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 实现 shuffle 功能
"""

import time
import threading
from init import *

# 线程池 (reduce)
shuffle_thread_pool = []
# 开始时间
start_time = 0


def judge_word(word: str) -> int:
    if 97 <= ord(word) <= 105 or 65 <= ord(word) <= 73:  # a(A) 到 i(I)
        return 1
    elif 106 <= ord(word) <= 114 or 74 <= ord(word) <= 82:  # j(J) 到 r(R)
        return 2
    else:
        return 3


def create_thread(idx: int) -> threading.Thread:
    """创建线程 (shuffler)

    Args:
        idx (int): 待创建的线程序号

    Returns:
        threading.Thread: 新创建的线程
    """
    file_dir = os.path.join(TMP_DIR, 'combine' + str(idx))
    shuffle_dir = os.path.join(TMP_DIR, 'shuffle')
    return threading.Thread(target=shuffler(file_dir, shuffle_dir), args=('t' + str(idx),), name='t' + str(idx))


def join_thread(idx: int, timer_on: bool = True) -> None:
    """阻塞线程 (shuffler)

    Args:
        idx (int): 待阻塞的线程序号
        timer_on (bool): 是否启用计时，默认启用
    """
    shuffle_thread_pool[idx - 1].join()
    if timer_on:
        print("t%d %s s." % (idx, time.perf_counter() - start_time))


def shuffler(src_file: str, dst_file: str):
    """shuffler 功能函数

    Args:
        src_file (str): 源文件路径
        dst_file (str): 目标文件路径
    """
    f_read = open(src_file, 'r')
    f_write1 = open(dst_file + '1', 'a')
    f_write2 = open(dst_file + '2', 'a')
    f_write3 = open(dst_file + '3', 'a')

    for line in f_read:
        line = line.strip()
        word, count = line.split(',', 1)
        idx = judge_word(word[0])
        if idx == 1:
            f_write1.write("{},{}\n".format(word, count))
        elif idx == 2:
            f_write2.write("{},{}\n".format(word, count))
        else:
            f_write3.write("{},{}\n".format(word, count))


class Shuffler(threading.Thread):
    def __init__(self, combine_queue, shuffle_node_count):
        super().__init__()
        self.combine_queue = combine_queue
        self.shuffle_node_count = shuffle_node_count
        self.cur_node_count = 0
        self.shuffle_thread_list = []
        self.lock = threading.Lock()

    def create_shuffle(self, idx: int):
        # print("shuffle thread %d start!" % idx)
        shuffle_thread = create_thread(idx)
        shuffle_thread.start()
        shuffle_thread.join()
        # * 发送消息，'shuffleX' 已就绪 (X = 1, 2, 3)
        # print("shuffle thread %d finish!" % idx)

    def run(self):
        while self.cur_node_count < self.shuffle_node_count:
            msg = self.combine_queue.get()
            if msg:
                idx = int(msg)
                shuffle_t = threading.Thread(
                    target=self.create_shuffle, args=(idx,))
                self.shuffle_thread_list.append(shuffle_t)
                shuffle_t.start()
                self.lock.acquire()
                self.cur_node_count += 1
                self.lock.release()

        for i in range(self.shuffle_node_count):
            self.shuffle_thread_list[i].join()
        print("shuffle threads all finish.")

# -*- coding: utf-8 -*-

"""
# @File       : shuffler.py
# @Time       : 2022/11/24
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 实现 shuffle 功能
"""

import os
import re
import time
import threading

SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
TMP_DIR = os.path.join(BASE_DIR, 'tmp')  # 临时文件目录

# 线程池 (reduce)
shuffle_thread_pool = []
# 开始时间
start_time = 0


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
        if word[0] == 'a' or word[0] == 'A' or word[0] == 'b' or word[0] == 'B' or word[0] == 'c' or word[0] == 'C':
            f_write1.write("{},{}\n".format(word, count))
        elif word[0] == 'd' or word[0] == 'D' or word[0] == 'e' or word[0] == 'E' or word[0] == 'f' or word[0] == 'F':
            f_write1.write("{},{}\n".format(word, count))
        elif word[0] == 'g' or word[0] == 'G' or word[0] == 'h' or word[0] == 'H' or word[0] == 'i' or word[0] == 'I':
            f_write1.write("{},{}\n".format(word, count))
        elif word[0] == 'j' or word[0] == 'J' or word[0] == 'k' or word[0] == 'K' or word[0] == 'l' or word[0] == 'L':
            f_write2.write("{},{}\n".format(word, count))
        elif word[0] == 'm' or word[0] == 'M' or word[0] == 'n' or word[0] == 'N' or word[0] == 'o' or word[0] == 'O':
            f_write2.write("{},{}\n".format(word, count))
        elif word[0] == 'p' or word[0] == 'P' or word[0] == 'q' or word[0] == 'Q' or word[0] == 'r' or word[0] == 'R':
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
        print("shuffle thread %d start!" % idx)
        shuffle_thread = create_thread(idx)
        shuffle_thread.start()
        shuffle_thread.join()
        # * 发送消息，'shuffleX' 已就绪 (X = 1, 2, 3)
        print("shuffle thread %d finish!" % idx)

    def run(self):
        while self.cur_node_count < self.shuffle_node_count:
            msg = self.combine_queue.get()
            # combine_match = re.findall(r'combine(\d+)', msg)
            # if combine_match:
            if msg:
                # idx = int(combine_match[0])
                idx = int(msg)
                shuffle_t = threading.Thread(
                    target=self.create_shuffle, args=(idx, ))
                self.shuffle_thread_list.append(shuffle_t)
                shuffle_t.start()
                self.lock.acquire()
                self.cur_node_count += 1
                self.lock.release()

        for i in range(self.shuffle_node_count):
            self.shuffle_thread_list[i].join()
        print("shuffle threads all finish.")


if __name__ == "__main__":
    for i in range(1, 10):
        shuffle_thread_pool.append(create_thread(i))

    start_time = time.perf_counter()

    for thread in shuffle_thread_pool:
        thread.start()

    for i in range(1, 10):
        join_thread(i)

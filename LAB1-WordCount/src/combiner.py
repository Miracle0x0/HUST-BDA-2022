# -*- coding: utf-8 -*-

"""
# @File       : combiner.py
# @Time       : 2022/11/24
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 实现 combine 功能
"""

import time
import threading
from init import *

# 线程池 (combine)
combine_thread_pool = []
# 开始时间
start_time = 0


def create_thread(idx: int) -> threading.Thread:
    """创建线程 (combine)

    Args:
        idx (int): 待创建的线程序号

    Returns:
        threading.Thread: 新创建的线程
    """
    map_dir = os.path.join(TMP_DIR, 'map' + str(idx))
    combine_dir = os.path.join(TMP_DIR, 'combine' + str(idx))
    return threading.Thread(target=combiner(map_dir, combine_dir), args=('t' + str(idx),), name='t' + str(idx))


def join_thread(idx: int, timer_on: bool = True) -> None:
    """阻塞线程 (combine)

    Args:
        idx (int): 待阻塞的线程序号
        timer_on (bool): 是否启用计时，默认启用
    """
    combine_thread_pool[idx - 1].join()
    if timer_on:
        print("t%d %s s." % (idx, time.perf_counter() - start_time))


def combiner(src_file: str, dst_file: str) -> None:
    """combine 功能函数

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

    count_dict = sorted(count_dict.items(), key=lambda x: x[0], reverse=False)

    for k, v in count_dict:
        f_write.write("{},{}\n".format(k, v))


class Combiner(threading.Thread):
    def __init__(self, map_queue, combine_queue, combine_node_count: int):
        super().__init__()
        self.map_queue = map_queue
        self.combine_queue = combine_queue
        self.combine_node_count = combine_node_count
        self.cur_node_count = 0
        self.combine_thread_list = []
        self.lock = threading.Lock()

    def create_combine(self, idx: int):
        # print("combine thread %d start!" % idx)
        combine_thread = create_thread(idx)
        combine_thread.start()
        combine_thread.join()
        # * 发送消息，'combineX' 已就绪 (X = 1, 2, ..., 9)
        # self.combine_queue.put('combine' + str(idx))
        self.combine_queue.put(str(idx))
        # print("combine thread %d finish!" % idx)

    def run(self):
        while self.cur_node_count < self.combine_node_count:
            msg = self.map_queue.get()
            if msg:
                idx = int(msg)
                combine_t = threading.Thread(
                    target=self.create_combine, args=(idx,))
                self.combine_thread_list.append(combine_t)
                combine_t.start()
                self.lock.acquire()
                self.cur_node_count += 1
                self.lock.release()

        for i in range(self.combine_node_count):
            self.combine_thread_list[i].join()
        print("combine threads all finish.")

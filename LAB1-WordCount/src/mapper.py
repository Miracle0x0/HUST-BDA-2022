# -*- coding: utf-8 -*-

"""
# @File       : mapper.py
# @Time       : 2022/11/24
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: 实现 map 功能
"""

import os
import threading
import time

SRC_DIR = os.path.dirname(__file__)  # 代码文件目录
BASE_DIR = os.path.dirname(SRC_DIR)  # 项目根目录
DATA_DIR = os.path.join(BASE_DIR, 'data')  # 数据文件目录
TMP_DIR = os.path.join(BASE_DIR, 'tmp')  # 临时文件目录
MAP_NODE_COUNT = 9  # map 节点个数

# 线程池 (map)
map_thread_pool = []
# 开始时间
start_time = 0


def read_file(file):
    """读取文件

    Args:
        file (TextIOWrapper): 待读取的文件

    Yields:
        str: 切分后的词组
    """
    for line in file:
        line = line.strip()
        yield line.split(', ')


def mapper(src_file: str, dst_file: str) -> None:
    """map 功能函数

    Args:
        src_file (str): 源文件路径
        dst_file (str): 目标文件路径
    """
    f_read = open(src_file, 'r')
    f_write = open(dst_file, 'w')
    lines = read_file(f_read)
    with f_write as f:
        for words in lines:
            for word in words:
                f.write("{},{}\n".format(word, 1))


def create_thread(idx: int) -> threading.Thread:
    """创建线程 (mapper)

    Args:
        idx (int): 待创建的线程序号

    Returns:
        threading.Thread: 新创建的线程
    """
    src_dir = os.path.join(DATA_DIR, 'source0' + str(idx))  # 源文件目录
    dst_dir = os.path.join(TMP_DIR, 'map' + str(idx))  # map 文件输出目录
    return threading.Thread(target=mapper(src_dir, dst_dir), args=('t' + str(idx),), name='t' + str(idx))


def join_thread(idx: int) -> None:
    """阻塞线程 (mapper)

    Args:
        idx (int): 待阻塞的线程序号
    """
    map_thread_pool[idx - 1].join()
    print("t%d: %s s" % (idx, time.perf_counter() - start_time))


class Mapper(threading.Thread):
    def __init__(self, map_queue, map_node_count: int):
        super().__init__()
        self.map_queue = map_queue
        self.map_node_count = map_node_count
        self.cur_node_count = 0
        self.map_thread_list = []
        self.lock = threading.Lock()

    def create_map(self, idx: int):
        print("map thread %d start!" % idx)
        map_thread = create_thread(idx)
        map_thread.start()
        map_thread.join()
        # * 发送消息，'mapX' 已就绪 (X = 1, 2, ..., 9)
        # self.map_queue.put('map' + str(idx))
        self.map_queue.put(str(idx))
        print("map thread %d finish!" % idx)

    def run(self):
        while self.cur_node_count < self.map_node_count:
            idx = self.cur_node_count + 1
            map_t = threading.Thread(target=self.create_map, args=(idx,))
            self.map_thread_list.append(map_t)
            map_t.start()
            self.lock.acquire()
            self.cur_node_count += 1
            self.lock.release()

        for i in range(self.map_node_count):
            self.map_thread_list[i].join()
        print("map threads all finish.")


if __name__ == "__main__":
    for i in range(1, MAP_NODE_COUNT + 1):
        map_thread_pool.append(create_thread(i))

    start_time = time.perf_counter()

    for thread in map_thread_pool:
        thread.start()

    for i in range(1, MAP_NODE_COUNT + 1):
        join_thread(i)

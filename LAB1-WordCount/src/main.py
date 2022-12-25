# -*- coding: utf-8 -*-

"""
# @File       : main.py
# @Time       : 2022/11/25
# @Author     : Asuna
# @Version    : Python 3.11
# @Description: LAB1 WordCount 实现
"""

import time
from queue import Queue

from mapper import Mapper
from reducer import Reducer
from shuffler import Shuffler
from combiner import Combiner
from results import generate_result

MAP_NODE_COUNT = 9  # map 节点个数
COMBINE_NODE_COUNT = 9  # combine 节点个数
REDUCE_NODE_COUNT = 3  # reduce 节点个数
SHUFFLE_NODE_COUNT = 9  # shuffle 节点个数

if __name__ == "__main__":
    start_time = time.time()

    map_queue = Queue(9)
    combine_queue = Queue(9)

    # * Map 节点组
    mapper = Mapper(map_queue=map_queue, map_node_count=MAP_NODE_COUNT)
    mapper.start()

    # * Combine 节点组
    combiner = Combiner(map_queue=map_queue, combine_queue=combine_queue,
                        combine_node_count=COMBINE_NODE_COUNT)
    combiner.start()

    # * Shuffle 节点组
    shuffler = Shuffler(combine_queue=combine_queue,
                        shuffle_node_count=SHUFFLE_NODE_COUNT)
    shuffler.start()

    # * Reduce 节点组
    reducer = Reducer(reduce_node_count=REDUCE_NODE_COUNT)

    mapper.join()
    combiner.join()
    shuffler.join()
    # print("Process done.")

    # print("Reduce start.")
    reducer.start()
    reducer.join()
    # print("Reduce done.")

    generate_result()

    print("All done.")

    finish_time = time.time()

    print("WordCount Algorithm time: %.3f s." % (finish_time - start_time))

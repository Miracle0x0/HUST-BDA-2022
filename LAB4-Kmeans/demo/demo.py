import numpy as np
import random

k = 3
centers = []
for i in range(k):
    c = []
    c.append(0)
    for j in range(13):
        c.append(random.random())
    centers.append(c)

print(len(centers), len(centers[0]))
demo = np.array(centers)
print(demo.shape)
print(demo)

hhh = np.random.random((3, 14))
print(hhh.shape)
hhh[:, 0] = 0
print(hhh)

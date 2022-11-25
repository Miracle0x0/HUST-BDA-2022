# 实验二 PageRank 算法及其实现

## 2.1 实验目的

1. 学习 pagerank 算法并熟悉其推导过程；
2. 实现 pagerank 算法；（可选进阶版）理解阻尼系数；
3. 将 pagerank 算法运用于实际，并对结果进行分析。

- 基本 pagerank 公式

$$
r = Mr
$$

- 进阶版 pagerank 公式

$$
r = \beta Mr + (1 - \beta)[\frac{1}{N}]_{N \times N}
$$

其中 $\beta$ 为阻尼系数，常见值为 0.85。

## 2.2 实验内容

提供的数据集包含邮件内容（emails.csv），人名与 id 映射（persons.csv），别名信息（aliases.csv），emails 文件中只考虑 MetadataTo 和 MetadataFrom 两列，分别表示**收件人**和**寄件人姓名**，但这些姓名包含许多别名，思考如何对邮件中人名进行统一并映射到唯一 id？（提供预处理代码 `preprocess.py` 以供参考）。

完成这些后，即可由寄件人和收件人为节点构造有向图，不考虑重复边，编写 pagerank 算法的代码，根据每个节点的入度计算其 pagerank 值，迭代直到误差小于 $10^{-8}$。

实验进阶版考虑加入 teleport $\beta$，用以对概率转移矩阵进行修正，解决 dead ends 和 spider trap 的问题。

输出人名 id 及其对应的 pagerank 值。

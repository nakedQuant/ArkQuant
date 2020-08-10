# -*- coding :utf-8 -*-
'''
    计算相关性、自相关性、偏相关性常见的算法
'''
import numpy as np

from algorithm import zoom

def Euclidean(x,y):
    """
        1 /（1 + 距离） y和y_fit的euclidean欧式距离(L2范数)、点与点之间的绝对距离
        扩展：
            1、 y和y_fit的manhattan曼哈顿距离(L1范数) 坐标之间的绝对距离之和
            2、 y和y_fit切比雪夫距离 max(各个维度的最大值)
    """
    x_scale = zoom(x)
    y_scale = zoom(y)
    distance = np.sqrt((x_scale - y_scale) ** 2)
    p = 1 / (1 + distance)
    return p

def CosDistance(x,y):
    """
        1、余弦相似度（夹角，90无关，0为相似度1） y和y_fit的cosine余弦距离（相似性） 向量之间的余弦值
    """
    x_s = zoom(x)
    y_s = zoom(y)
    cos = x.y / (np.sqrt(x_s ** 2) * np.sqrt(y_s ** 2))
    return cos

def CovDistance(x,y):
    """
        1、基于相关系数数学推导， 协方差与方差 马氏距离 将数据投影到N(0, 1)  区间并求其欧式距离，称为数据的协方差距离
    """
    x_s = zoom(x)
    y_s = zoom(y)
    cov = (x * y).mean() - (x_s.mean()) * y_s.mean()
    p = cov / (x.std() * y.std())
    return p

def SignDistance(x,y):

    """
        1、符号＋－相关系数, sign_x = np.sign(x), sign_y = np.sign(y), np.corrcoef(sign_x, sign_y)[0][1]
    """
    sign_x = np.sign(x)
    sign_y = np.sign(y)
    p = (sign_x * sign_y).sum() / len(x)
    return  p

def RankDistance(x,y):
    """
        1、排序rank(ascending=False, method='first') 计算相关系数
    """
    x_rank = x.rank()
    y_rank = y.rank()
    p = CovDistance(x_rank,y_rank)
    return p
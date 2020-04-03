# -*- coding : utf-8 -*-
import numpy as np

from Algorithm.Feature import BaseFeature

class PCA(BaseFeature):
    '''
    主成分分析法理论：选择原始数据中方差最大的方向，选择与其正交而且方差最大的方向，不断重复这个过程
    pca.fit_transform()
    具体的算法：
    PCA算法：
    1）将原始数据按列组成n行m列矩阵X

    2）将X的每一行（代表一个属性字段）进行零均值化，即减去这一行的均值

    3）求出协方差矩阵C=X * XT

    4）求出协方差矩阵的特征值及对应的特征向量

    5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵P

    6）Y=PX  即为降维到k维后的数据
    '''
    _topNfeat = 2

    @classmethod
    def calc_feature(cls,feed):
        datamat = feed.copy()
        meanval = np.mean(datamat, axis=0)
        meanremoved = datamat - meanval
        print('mean',meanremoved)
        covmat = np.cov(meanremoved, rowvar=0)
        eigval, eigvect = np.linalg.eig(np.mat(covmat))
        eigvalind = np.argsort(eigval)
        eigvalind = eigvalind[-cls._topNfeat :]
        redeigvect = eigvect[:, eigvalind]
        reconmat = meanremoved * redeigvect * redeigvect.T + meanval
        return reconmat
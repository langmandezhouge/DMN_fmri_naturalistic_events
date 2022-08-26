#pearson相关分析
#方式一：两个向量之间做相关，np.corrcoef函数.两个矩阵可以先拉成向量，再做相关
#方式二：一个矩阵内列与列之间做相关（事件关系矩阵）

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import stats
import os

input_dir = '/prot/lkz/searchlihgt_pearson/pearson/results/event_result/0.npy'
dataset = np.load(input_dir,allow_pickle=True)
print(dataset.shape)

#方式一：两个向量之间做相关
'''pearson = np.corrcoef(x, y)  #x和y为两个向量.若为矩阵，则先将x和y拉成一维向量，再做相关
print(pearson.shape)
print(pearson)'''


#方式二：一个矩阵内列与列之间做相关（即构建事件关系矩阵）
data = np.transpose(dataset) # 将得到的每一个体素的所有事件数组转置为（27，m），列为事件数m
print(data.shape)

#pearson correlation analysis
data = pd.DataFrame(data)
pearson = data.corr() #对每个体素的所有事件数组，列与列之间做相关
print(pearson.shape)
print(pearson)

import seaborn as sb
import matplotlib.pyplot as plt
sb.heatmap(data = pearson)
#sb.heatmap(data = pearson,cmap="YlGnBu") # cmap为不同的颜色背景
plt.show()

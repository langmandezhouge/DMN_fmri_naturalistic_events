# -*- coding: utf-8 -*-

#pearson相关分析
#方式一：两个向量之间做相关，np.corrcoef函数.两个矩阵可以先拉成向量，再做相关
#方式二：一个矩阵内列与列之间做相关（事件关系矩阵）

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import stats
import os
import seaborn as sb
import matplotlib.pyplot as plt

#方式一：两个向量之间做相关
'''pearson = np.corrcoef(x, y)  #x和y为两个向量.若为矩阵，则先将x和y拉成一维向量，再做相关
print(pearson.shape)
print(pearson)'''


#方式二：一个矩阵内列与列之间做相关（即构建事件关系矩阵）

path = '/prot/lkz/searchlihgt_pearson/pearson/results/voxel_events_result/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[:-4]))
print(files)

count = len(files)
print(count)

for i in range(count):
    datas = files[i]
    dataset = np.load(path + datas)
    print(dataset.shape)
    data = np.transpose(dataset) # zhuanzhi
    print(data.shape)
    print(data)
    data = pd.DataFrame(data)

    pearson = data.corr()
    print(pearson.shape)
    print(pearson)
    #data.corr(method='spearman') #other method
    #data.corr(method='kendall')

    output_dir = '/prot/lkz/searchlihgt_pearson/pearson/results/events_pearson_corr_result/'
    np.save(os.path.join(output_dir, str(i)), pearson)

    heatmap = sb.heatmap(pearson)
    #sb.heatmap(data = pearson,cmap="YlGnBu")
    plt.show()

    heatmap = heatmap.get_figure()
    heatmap.savefig(os.path.join(output_dir, str(i)), dpi = 300)

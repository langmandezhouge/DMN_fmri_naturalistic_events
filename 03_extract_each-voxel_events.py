# -*- coding: utf-8 -*-

#将所有事件中同一个体素的向量提取出来，并保存为一个数组（m，27），m为事件个数，共得到n（总体素数）个数组
#最终得到每一个体素的所有事件向量的矩阵

import numpy as np
import os
import pandas as pd

path = '/prot/lkz/searchlihgt_pearson/pearson/results/Events/'

files = os.listdir(path) #遍历保存的事件数组路径下所有的文件
files.sort(key=lambda x:int(x[:-4])) #os.listdir并不是按照顺序，因此将其整成按照顺序来取所有文件
print(files)

m = len(files) #事件个数m
print(m)

output_dir = '/prot/lkz/searchlihgt_pearson/pearson/results/event_result/'


for i in range(n):
    df = pd.DataFrame()
    for j in range(m):
        datas = files[j] #第j个事件
        datas = np.load(path + datas)
        data = datas[i] #第j个事件中第i个体素向量（27，1）
       # print(data)
        print(data.shape)
        data = pd.DataFrame(data)
        data = np.transpose(data)  # 转置函数变成（1，27）
        df = df.append(data)
    print(df.shape)
    print(df)
    np.save(os.path.join(output_dir, str(i)), df) #将每一个体素的所有事件数组保存为对应体素序号的文件


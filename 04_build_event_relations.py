# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import stats
import os
import seaborn as sb
import matplotlib.pyplot as plt


path = '/prot/lkz/DMN_fmri_naturalistic_events/results/03_each-voxel_event-matrix/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[0:-17]))
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

    output_dir = '/prot/lkz/DMN_fmri_naturalistic_events/results/04_each-voxel_event-relation/'
    np.save(os.path.join(output_dir, str(i+1)+'_event_relations'), pearson)

'''    heatmap = sb.heatmap(pearson)
    #sb.heatmap(data = pearson,cmap="YlGnBu")
    plt.show()

    heatmap = heatmap.get_figure()
    heatmap.savefig(os.path.join(output_dir, str(i+1)+'_event_relations'), dpi = 300)'''

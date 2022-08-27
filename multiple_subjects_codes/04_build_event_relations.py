# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import stats
import os
import seaborn as sb
import matplotlib.pyplot as plt


path = '/prot/lkz/searchlihgt_pearson/subjects/'

files = os.listdir(path)
##files.sort(key=lambda x:int(x[0:-17]))
print(files)

count = len(files)
print(count)

for i in range(count):
    subj_path = path + 'sub-' + '%.2d' % (i + 1) + '/' + 'results' + '/' + '03_each-voxel_event-matrix' + '/'
    file = os.listdir(subj_path)
    num = len(file)
    for j in range(num):
        voxel_events_path = subj_path  + '%.5d' % (j + 1) + '_voxel_events.npy'
        voxel_events = voxel_events_path
        dataset = np.load(voxel_events)
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

        output_dir = path + 'sub-' + '%.2d' % (i + 1) + '/' + 'results' + '/' + '04_each-voxel_event-relation' + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, '%.5d' % (j + 1)+'_event_relations'), pearson)

'''        heatmap = sb.heatmap(pearson)
        #sb.heatmap(data = pearson,cmap="YlGnBu")
        plt.show()

        heatmap = heatmap.get_figure()
        heatmap.savefig(os.path.join(output_dir, '%.5d' % (j + 1)+'_event_relations'), dpi = 300)'''

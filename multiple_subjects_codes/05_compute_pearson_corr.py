# compute the Pearson correlation coefficient between two vectors or matrices

import numpy as np
import pandas as pd
import os

path = '/prot/lkz/searchlihgt_pearson/subjects/'

files = os.listdir(path)
#files.sort(key=lambda x: int(x[0:-17]))
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
        datas = np.load(voxel_events)
        print(datas.shape)
        a = datas.shape[0]
        b = datas.shape[1]
        data = datas.reshape(1, a * b)
        datas = pd.DataFrame(datas)

        target_path = '/prot/lkz/DMN_fmri_naturalistic_events/results/target.npy'
        target = np.load(target_path)
        m = target.shape[0]
        n = target.shape[1]
        target = target.reshape(1, m * n)
        target = pd.DataFrame(target)

        pearson_corr = np.corrcoef(data, target)
        print(pearson_corr.shape)
        print(pearson_corr)

        #heatmap = sb.heatmap(pearson)
        #sb.heatmap(data = pearson,cmap="YlGnBu")
        #plt.show()

        output_dir = path + 'sub-' + '%.2d' % (i + 1) + '/' + 'results' + '/' + '05_matrixs_pearon-corr' + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, '%.5d' % (j + 1) + '_pearson_corr'), pearson_corr)

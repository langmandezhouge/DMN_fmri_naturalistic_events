# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

path = '/prot/lkz/searchlihgt_pearson/subjects/'

files = os.listdir(path)
#files.sort(key=lambda x: int(x[1:-7]))
print(files)

count = len(files)
print(count)

for i in range(count):
    subj_path = path + 'sub-' + '%.2d' % (i + 1) + '/' + 'results' + '/' + '02_searchlight_events-matrix' + '/'
    file = os.listdir(subj_path)
    num = len(file)
    voxel_file = subj_path + 'searchlight_E01.npy'
    voxel_file = np.load(voxel_file)
    n = voxel_file.shape[0]

    for k in range(n):
        print(k)
        df = pd.DataFrame()
        for j in range(num):
            voxel_events_path = subj_path  + 'searchlight_E' + '%.2d' % (j + 1) + '.npy'
            voxel_events = voxel_events_path
            datas = np.load(voxel_events)
            data = datas[k]
            # print(data)
            print(data.shape)
            data = pd.DataFrame(data)
            data = np.transpose(data)
            df = df.append(data)
        print(df.shape)
        print(df)

        output_dir = path + 'sub-' + '%.2d' % (i + 1) + '/' + 'results' + '/' + '03_each-voxel_event-matrix' + '/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, '%.5d' % (k + 1) + '_voxel_events'), df)

# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd

path = '/prot/lkz/DMN_fmri_naturalistic_events/results/02_searchlight_events-matrix/'

files = os.listdir(path) 
files.sort(key=lambda x:int(x[13:-4]))
print(files)

file = files[0]
file = np.load(path + file)
n = file.shape[0]

m = len(files) 
print(m)

output_dir = '/prot/lkz/DMN_fmri_naturalistic_events/results/03_each-voxel_event-matrix/'


for i in range(n):
    df = pd.DataFrame()
    for j in range(m):
        datas = files[j]         
        datas = np.load(path + datas)
        data = datas[i]       
        # print(data)
        print(data.shape)
        data = pd.DataFrame(data)
        data = np.transpose(data)        
        df = df.append(data)
    print(df.shape)
    print(df)
    np.save(os.path.join(output_dir, str(i+1) + '_voxel_events'), df) 

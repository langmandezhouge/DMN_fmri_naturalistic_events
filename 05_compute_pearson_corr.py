#compute the Pearson correlation coefficient between two vectors or matrices

import numpy as np
import pandas as pd
import os

path = '/prot/lkz/DMN_fmri_naturalistic_events/results/03_each-voxel_event-matrix/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[0:-17]))
print(files)

count = len(files)
print(count)

target_path = '/prot/lkz/DMN_fmri_naturalistic_events/results/target.npy'
target = np.load(target_path)
m = target.shape[0]
n = target.shape[1]
target = target.reshape(1,m*n)
target = pd.DataFrame(target) 
                 
for i in range(count):
    data = files[i]
    datas = np.load(path + data)
    print(datas.shape)
    a = datas.shape[0]
    b = datas.shape[1]
    data = datas.reshape(1,a*b)
    datas = pd.DataFrame(datas) 
    
    pearson_corr = np.corrcoef(data, target)  
    print(pearson_corr.shape)
    print(pearson_corr)
        
#    heatmap = sb.heatmap(pearson)
    #sb.heatmap(data = pearson,cmap="YlGnBu")
#    plt.show()
                 
    output_dir = '/prot/lkz/DMN_fmri_naturalistic_events/results/05_matrixs_pearon-corr/'
    np.save(os.path.join(output_dir, str(i+1)+'_pearson_corr'), pearson_corr)

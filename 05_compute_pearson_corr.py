#compute the Pearson correlation coefficient between two vectors or matrices

import numpy as np
import pandas as pd
import os

path = '/prot/lkz/searchlihgt_pearson/pearson/results/voxel_events_result/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[:-4]))
print(files)

count = len(files)
print(count)

target_path = '/prot/lkz/searchlihgt_pearson/pearson/results/target.npy'
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
                 
    output_dir = '/prot/lkz/searchlihgt_pearson/pearson/results/pearson_corr_result/'
    np.save(os.path.join(output_dir, str(i)), pearson_corr)         

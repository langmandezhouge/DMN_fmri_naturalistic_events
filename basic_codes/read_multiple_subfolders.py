import os, sys
import numpy as np

path = '/prot/lkz/searchlihgt_pearson/pearson/results/voxel_events_result/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[4:]))
print(files)

count = len(files)
print(count)

for i in range(count):
    subj_path = path + 'subj' + '%.2d' % (i+1) + '/' + '%.2d' % (i+1) + '.npy'
    data = subj_path
    data = np.load(data)
  #  print(data)
    print(data.shape)
    datas = data.reshape(1,81)
    print(datas.shape)

    output_dir = path + 'subj' + '%.2d' % (i+1) + '/' + 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_dir + '/' + '%.2d' % (i+1) + '.npy',datas)

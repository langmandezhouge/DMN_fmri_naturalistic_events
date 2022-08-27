import numpy as np

dataset = np.load('/prot/lkz/searchlihgt_pearson/pearson/results/voxel_events_result/subj01/results/01.npy',allow_pickle=True)
print(dataset)
#print(np.nonzero(dataset))
print(dataset.shape)

import numpy as np

dataset = np.load('/prot/lkz/searchlight/Naturalistic/results/dataset_4d.npy')
dataset_2d = dataset.reshape(dataset.shape[0]*dataset.shape[1]*dataset.shape[2],dataset.shape[3])
#dataset_2d = np.reshape(dataset,(dataset.shape[0]*dataset.shape[1]*dataset.shape[2],dataset.shape[3]))
print(dataset_2d)
print(dataset_2d.shape)

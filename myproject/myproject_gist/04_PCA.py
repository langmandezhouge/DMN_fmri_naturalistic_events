#PCA for the mean bold of story(across subjects),then extract events.

from sklearn.decomposition import PCA
import nibabel as nib
import numpy as np


path = '/prot/lkz/LSTM/roi_results/roi_story_mean/region-001/21styear/roi-001_21styear_mean_bold.npy'
data = np.load(path)
data = np.transpose(data)
pca = PCA(n_components=20) #n_components= 'mle'
bold_pca = pca.fit_transform(data)

print('Original data shape:', data.shape)
print('PCA data shape:', bold_pca.shape)

#PCA for the mean bold of story(across subjects),then extract events.

from sklearn.decomposition import PCA
import nibabel as nib
import numpy as np
import os
path = '/prot/lkz/LSTM/roi_results/roi_story_mean/'

for i in os.listdir(path):
    files = path + i
    f1 = files[-3:-1]
    f2 = files[-1]
    for j in os.listdir(files):
        file = files + "/" +j + "/" + "roi-" + f1 + f2 + "_" + j + "_mean_bold.npy"

        data = np.load(file)
        data = np.transpose(data)
        pca = PCA(n_components=60) #n_components= 'mle'
        bold_pca = pca.fit_transform(data)

        print('Original data shape:', data.shape)
        print('PCA data shape:', bold_pca.shape)

        print("save pca result: ", i + ":" + j)
        num = f1 + f2
        output = '/prot/lkz/my_project/my_project-gist/03_story_mean_pca_results/' + "region-" + num + "/" + j +"/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, "pca_" + j), bold_pca)

        
'''path = '/prot/lkz/LSTM/roi_results/roi_story_mean/region-216/'

for j in os.listdir(path):
        file = path + "/" +j + "/" + "roi-216_" + j + "_mean_bold.npy"
        data = np.load(file)
        data = np.transpose(data)
        pca = PCA(n_components=60) #n_components= 'mle'
        bold_pca = pca.fit_transform(data)

        print('Original data shape:', data.shape)
        print('PCA data shape:', bold_pca.shape)

        print("save pca result: ", "021" + ":" + j)
        output = '/prot/lkz/my_project-gist/03_story_mean_pca_results/' + "region-216/" + j +"/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, "pca_" + j), bold_pca)'''

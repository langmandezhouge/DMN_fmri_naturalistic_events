from sklearn.decomposition import PCA
import nibabel as nib
import numpy as np
import os
path = '/protNew/lkz/my_project/Narrative-result/twobrain_allbold_sub_mean/'

for i in os.listdir(path):
    region_path = path + i + "/"
    r = i[-3:]
    for j in os.listdir(region_path):
        file = region_path + "/" + j + "/" + "roi-" + r + "_" + j + "_mean_bold.npy"
        data = np.load(file)
        data = np.transpose(data)
        pca = PCA(n_components=60) #n_components= 'mle'
        bold_pca = pca.fit_transform(data)

        print("save pca result: ", i + ":" + j)
        output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/03_story_mean_pca_results/' + i + "/" + j +"/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, "pca_" + j), bold_pca)

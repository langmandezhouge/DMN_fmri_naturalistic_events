import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_result/rsa_p/'
files = os.listdir(path)
files.sort(key=lambda x:int(x[7:10]))
story_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/03_story_mean_pca_results/region-022/'


for i in os.listdir(story_path):
    datas = []
    for j in files:
        file_path = path + j + "/" + "rsa-" + i + ".npy"
        file_data = np.load(file_path)
        datas.append(file_data)

    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_story/p/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), datas)

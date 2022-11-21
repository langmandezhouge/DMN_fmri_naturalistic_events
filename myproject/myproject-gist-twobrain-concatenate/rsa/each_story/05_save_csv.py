import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_story/r/'

for i in os.listdir(path):
    j = i[0:-4]
    file = path + i
    dataset = np.load(file,allow_pickle=True)
    data = pd.DataFrame(dataset)
    out = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_story_csv/r/' + j
    data.to_csv(out + '.csv')

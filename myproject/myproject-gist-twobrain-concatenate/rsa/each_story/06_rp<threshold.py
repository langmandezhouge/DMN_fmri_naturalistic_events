import os
import numpy as np

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_story/p/'
for i in os.listdir(path):
    k = i[0:-4]
    file_path = path + i
    file = np.load(file_path)
    data = []
    for j, row in enumerate(file):
        if row < 0.05:
            row = 1
        else:
            row = 0
        data.append(row)

    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_story/p_0.05/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, k), data)

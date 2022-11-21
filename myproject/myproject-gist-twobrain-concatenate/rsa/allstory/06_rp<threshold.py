import os
import numpy as np

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/p_allstory.npy'
file = np.load(path)
data = []
for j, row in enumerate(file):
    if row < 0.05:
        row = 1
    else:
        row = 0
    data.append(row)

output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/'
np.save(os.path.join(output, "p_0.05_allstory"), data)

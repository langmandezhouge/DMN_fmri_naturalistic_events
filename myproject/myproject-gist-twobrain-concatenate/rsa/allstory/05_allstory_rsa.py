import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/rsa_p/'
#path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/rsa_result/rsa_result/rsa_r/'
files = os.listdir(path)
files.sort(key=lambda x:int(x[4:7]))

datas = []
for i in files:
    file_path = path + i
    file_data = np.load(file_path)
    datas.append(file_data)

    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/'
    #output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/rsa_result/rsa_result/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, "r_6story"), datas)

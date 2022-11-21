import os
import numpy as np
import pandas as pd

#path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/text_events/'
path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/text_events/text_events/'

files = os.listdir(path)
files.sort()

df = pd.DataFrame()
for i in files:
    file_path = path + i
    file = np.load(file_path)
    datas = pd.DataFrame(file)
    df = df.append(datas)

    #output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/'
    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, "text_events"), df)

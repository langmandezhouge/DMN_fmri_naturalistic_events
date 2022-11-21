import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/03_story_mean_pca_results/'
files = os.listdir(path)
files.sort(key=lambda x:int(x.split('-')[-1]))

#skip_tasks = ['21styear','forgot','lucy','merlin','notthefallintact','original','piemanpni','shapesphysical','synonyms', 'tunnel', 'vodka']
for i in files:
    file = path + i + "/"
    list4 = []
    df = pd.DataFrame()
    region_file = os.listdir(file)
    region_file.sort()
    print(region_file)
    for j in region_file:
        '''if j in skip_tasks:
            print(f"Skipping {j} ")
            continue'''
        pca_file = file + j + "/" + "pca_" + j + ".npy"
        data = np.load(pca_file)
        #time_path = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/07_mid-scale_times/' + j + ".npy"
        time_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/times/' + j + ".npy"
        time_file = np.load(time_path)
        num = len(time_file)
        list1 = []
        list2 = []
        list3 = []
        for k in range(num):
            onset  = time_file[k][0]
            offset = time_file[k][1] + 1
            bold_event = data[onset:offset, :]
            data_mean = np.mean(bold_event, axis=0)
            datas = pd.DataFrame(data_mean)
            datas = np.transpose(datas)
            df = df.append(datas)

    print("save pca_event_story result: ", i + ":" + j)
    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/bold_events/'
    #output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/bold_events/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), df)

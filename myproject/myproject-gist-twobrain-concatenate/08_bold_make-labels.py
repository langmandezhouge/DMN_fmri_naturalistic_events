import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/03_story_mean_pca_results/'
event_path = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/08_mid-scale_events/'
files = os.listdir(path)
files.sort(key=lambda x:int(x.split('-')[-1]))

for i in files:
    file = path + i + "/"
    list4 = []
    for j in os.listdir(file):
        pca_file = file + j + "/" + "pca_" + j + ".npy"
        data = np.load(pca_file)
        time_path = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/07_mid-scale_times/' + j + ".npy"
        time_file = np.load(time_path)
        num = len(time_file)
        event_file = event_path + j + ".npy"
        event_file = np.load(event_file, allow_pickle=True)
        list1 = []
        list2 = []
        list3 = []
        for k in range(num):
            onset  = time_file[k][0]
            offset = time_file[k][1] + 1
            bold_event = data[onset:offset, :]
            list1.append(bold_event)
            text_event = event_file[k]
            list2.append(text_event)
            time = time_file[k]
            list3.append(time)
            list4.append([bold_event, text_event])

        output1 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/09_pca_events_bold/' + i +"/"
        if not os.path.exists(output1):
            os.makedirs(output1)
        output2 = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/09_mid-scale_events_final/'
        if not os.path.exists(output2):
            os.makedirs(output2)
        output3 = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/09_mid-scale_times_final/'
        if not os.path.exists(output3):
            os.makedirs(output3)
        np.save(os.path.join(output1, j), list1)
        np.save(os.path.join(output2, j), list2)
        np.save(os.path.join(output3, j), list3)

    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/' + "10_labels/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i + "_labels"), list4)

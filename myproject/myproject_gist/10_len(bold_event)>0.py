import numpy as np
import os

path = '/prot/lkz/my_project/my_project-gist/'
time_path = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/07_mid-scale_times_final/'
bold_path = '/prot/lkz/my_project/my_project-gist/09_pca_events_bold/'
event_path = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/08_mid-scale_events_final/'

for i in os.listdir(bold_path):
    bold_file = bold_path + i +"/"
    list1 = []
    for j in os.listdir(bold_file):
        bold_sub_path = bold_file + j
        bold_sub_file = np.load(bold_sub_path,allow_pickle=True)
        event_file = event_path + j
        event_file = np.load(event_file, allow_pickle=True)
        time_file = time_path + j
        time_file = np.load(time_file, allow_pickle=True)

        list2 = []
        list3 = []
        list4 = []
        for k, data in enumerate(bold_sub_file):
            if len(data) > 0:
               bold  = bold_sub_file[k]
               list3.append(bold)
               event = event_file[k]
               list4.append(event)
               list1.append([bold, event])
               time = time_file[k]
               list2.append(time)

        np.save(os.path.join(time_path, j), list2)
        np.save(os.path.join(bold_file, j), list3)
        np.save(os.path.join(event_path, j), list4)
    output = path + "10_labels/"
    np.save(os.path.join(output, i + "_labels"), list1)


'''path = '/prot/lkz/my_project/my_project-gist/'
time_path = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/07_mid-scale_times_final/'
bold_path = '/prot/lkz/my_project/my_project-gist/09_pca_events_bold/region-100/'
event_path = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/08_mid-scale_events_final/'
list1 = []

for i in os.listdir(bold_path):
    bold_file = bold_path + i
    bold_file = np.load(bold_file,allow_pickle=True)
    event_file = event_path + i
    event_file = np.load(event_file, allow_pickle=True)
    time_file = time_path + i
    time_file = np.load(time_file, allow_pickle=True)

    list2 = []
    list3 = []
    list4 = []
    for j, data in enumerate(bold_file):
        if len(data) > 0:
           bold  = bold_file[j]
           list3.append(bold)
           event = event_file[j]
           list4.append(event)
           list1.append([bold, event])
           time = time_file[j]
           list2.append(time)


    np.save(os.path.join(time_path, i), list2)
    np.save(os.path.join(bold_path, i), list3)
    np.save(os.path.join(event_path, i), list4)
output = path + "10_labels/"
np.save(os.path.join(output,  "region-100_labels"), list1)'''

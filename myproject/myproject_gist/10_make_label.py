import os
import numpy as np

path = '/prot/lkz/my_project/my_project-gist/'
text_path = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/08_mid-scale_events_final/'
region_path = '/prot/lkz/my_project/my_project-gist/09_pca_events_bold/'

region_file = os.listdir(region_path)
region_file.sort(key=lambda x:int(x.split('-')[-1]))

skip_tasks = ['reach', 'slumlord']
for i in region_file:
    file = region_path + i + "/"
    data = []
    for j in os.listdir(file):
        if j in skip_tasks:
            print(f"Skipping {j} for events")
            continue
        bold_file_path = file + j
        bold_event = np.load(bold_file_path,allow_pickle=True)
        num = len(bold_event)
        text_file_path = text_path + j
        text_event = np.load(text_file_path,allow_pickle=True)
        for k in range(num):
            bold_label = bold_event[k]
            text_label = text_event[k]
            data.append([bold_label, text_label])

    output = path + "10_labels/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, "labels"), data)

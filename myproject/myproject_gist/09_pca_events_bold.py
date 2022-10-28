import os
import numpy as np

path = '/prot/lkz/my_project-gist/story_mean_pca_results/'
files = os.listdir(path)
files.sort(key=lambda x:int(x.split('-')[-1]))

skip_tasks = ['reach', 'slumlord']
for i in files:
    file = path + i + "/"
    for j in os.listdir(file):
        if j in skip_tasks:
            print(f"Skipping {j} for events")
            continue
        pca_file = file + j + "/" + "pca_" + j + ".npy"
        data = np.load(pca_file)
        time_path = '/prot/lkz/LSTM/text-gist/text_gist-results/07_mid-scale_times_final/' + j + ".npy"
        time_file = np.load(time_path)
        num = len(time_file)
        dataset = []
        for k in range(num):
            onset  = time_file[k][0]
            offset = time_file[k][1]
            data_event = data[onset:offset,:]
            dataset.append(data_event)

        print("save pca_event_story result: ", i + ":" + j)
        output = '/prot/lkz/my_project-gist/pca_events_bold/' + i + "/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, j), dataset)

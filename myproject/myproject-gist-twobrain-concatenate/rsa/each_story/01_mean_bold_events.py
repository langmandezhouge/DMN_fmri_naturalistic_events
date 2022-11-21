import os
import numpy as np
import pandas as pd

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/03_story_mean_pca_results/'
files = os.listdir(path)
files.sort(key=lambda x:int(x.split('-')[-1]))
text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/text_events/text_events/'

skip_tasks = ['reach', 'slumlord']
for i in files:
    file = path + i + "/"
    story = os.listdir(file)
    story.sort()
    for j in story:
        if j in skip_tasks:
            print(f"Skipping {j} for events")
            continue
        pca_file = file + j + "/" + "pca_" + j + ".npy"
        data = np.load(pca_file)
        time_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/text-gist/text_gist-results/09_mid-scale_times_final/' + j + ".npy"
        time_file = np.load(time_path)
        num = len(time_file)
        df = pd.DataFrame()
        for k in range(num):
            onset = time_file[k][0]
            offset = time_file[k][1] + 1
            bold_event = data[onset:offset, :]
            data_mean = np.mean(bold_event, axis=0)
            datas = pd.DataFrame(data_mean)
            datas = np.transpose(datas)
            df = df.append(datas)

        print("save pca_event_story result: ", i + ":" + j)
        output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/bold_events/' + i + "/"
        if not os.path.exists(output):
           os.makedirs(output)
        np.save(os.path.join(output, j), df)

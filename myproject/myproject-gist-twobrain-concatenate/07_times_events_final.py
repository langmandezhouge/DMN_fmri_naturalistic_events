import os
import pickle
import numpy as np

path = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/'
files = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/stimuli/'
event_vector = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/06_times_events-vector/'

for i in os.listdir(files):
    filename = path + "05_events_sent_times/" + i + ".txt"
    file = open(filename, "rb")
    file = pickle.load(file)
    num = len(file)
    event_path = event_vector + i + ".npy"
    event_file = np.load(event_path)
    data = []
    event_data = []
    for j in range(num):
        time = file[j]
        begin = time[0]
        endin = time[1]
        if endin >= begin:
            print(j)
            data.append(time)
            event = event_file[j]
            event_data.append(event)

    output = path + "07_mid-scale_times/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), data)
    output = path + "08_mid-scale_events/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), event_data)

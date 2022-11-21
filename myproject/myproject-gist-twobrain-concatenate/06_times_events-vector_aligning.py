import pickle
import os
import numpy as np

path = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/'
files = '/protNew/lkz/my_project/my_project-gist/onebrain/text-gist/text_gist-results/stimuli/'

for i in os.listdir(files):
    time_path = path + "05_events_sent_times/" + i + ".txt"
    event_path = path + "vector_result/" + i + ".txt"
    time_file = open(time_path, "rb")
    time_file = pickle.load(time_file)
    num = len(time_file)

    event_file = open(event_path, "rb")
    event_file = pickle.load(event_file)
    aligning_event_file = event_file[:num]
    print(aligning_event_file.shape)

    output = path + "06_times_events-vector/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), aligning_event_file)

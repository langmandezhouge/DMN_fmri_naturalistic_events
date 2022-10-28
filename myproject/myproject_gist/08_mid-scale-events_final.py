import os
import pickle
import numpy as np

path = '/prot/lkz/LSTM/text-gist/text_gist-results/'
files = '/prot/lkz/LSTM/text-gist/text_gist-results/stimuli/'

for i in os.listdir(files):
    event_vector = path + "06_times_events-vector/" + i + ".npy"
    event_vector = np.load(event_vector)
    filename = path + "05_events_sent_times/" + i + ".txt"
    file = open(filename, "rb")
    file = pickle.load(file)
    num = len(file)
    data = []
    for j in range(num):
        time = file[j]
        begin = time[0]
        endin = time[1]
        if endin > begin:
           final_event_vector = event_vector[j]
           data.append(final_event_vector)

    output = path + "08_mid-scale_events_final/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), data)

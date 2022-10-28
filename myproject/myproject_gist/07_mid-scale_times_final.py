import os
import pickle
import numpy as np

path = '/prot/lkz/LSTM/text-gist/text_gist-results/'
files = '/prot/lkz/LSTM/text-gist/text_gist-results/stimuli/'

for i in os.listdir(files):
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
            print(j)
            data.append(time)

    output = path + "07_mid-scale_times_final/"
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, i), data)

import math
import os
import csv
import pickle

path = '/prot/lkz/LSTM/text-gist/text_gist-results/'
files = '/prot/lkz/LSTM/text-gist/text_gist-results/stimuli/'

banned = ["'bout"]
for i in os.listdir(files):
    filename1 = path + "sent/" + i + ".txt"
    filename3 = path + "events_times_final/" + i + ".txt"
    filename2 = path + "stimuli/" + i + "/align.csv"
    list1 = []
    list2 = []
    list3 = []
    with open(filename1, "r", encoding="ascii") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            tmpt = line.replace(".", " ")\
                    .replace(",", " ")\
                    .replace("-", " ")\
                    .replace("?", " ")\
                    .replace(":", " ")\
                    .replace("!", " ")\
                    .replace(";", " ")\
                    .replace("(", " ")\
                    .replace(")", " ")\
                    .replace("--", " ")\
                    .replace("\"", " ")\
                    .split()
            for j in tmpt:
                if j.startswith("'") and j not in banned:
                    j = j[1:]
                if j.endswith("'"):
                    j = j[:-1]
                if len(j) > 0:
                    list1.append([j, idx])
    with open(filename2, "r", encoding="gb18030") as file:
        lines = csv.reader(file)
        for line in lines:
            list2.append([line[0].replace("’", "'").replace("\xf1", "n"), None if len(line) < 3 or len(line[2]) == 0 else line[2], None if len(line) < 4 or len(line[3]) == 0 else line[3]])
    begin = None
    endin = None
    count = -1
    list4 = []
    for j in range(len(list2)):
        if count != list1[j][1]:
            if count >= 0:
                if begin is not None and endin is not None:
                   if float(endin) - float(begin) >= 2.0:
                       begin_TR = math.ceil(float(begin) / 1.5)
                       endin_TR = math.floor(float(endin) / 1.5)
                       if endin_TR > begin_TR:
                           list4.append([begin_TR, endin_TR])
                else:
                    list4.append([])
            count = list1[j][1]
            begin = None
            endin = None
        if begin is None and list2[j][1] is not None:
            begin = list2[j][1]
        if begin is None and list2[j][2] is not None:
            begin = list2[j][2]
        if list2[j][1] is not None:
            endin = list2[j][1]
        if list2[j][2] is not None:
            endin = list2[j][2]
    if count >= 0:
        if begin is not None and endin is not None:
            if float(endin) - float(begin) >= 2.0:
                begin_TR = math.ceil(float(begin) / 1.5)
                endin_TR = math.floor(float(endin) / 1.5)
                if endin_TR > begin_TR:
                    list4.append([begin_TR, endin_TR])
        else:
            list4.append([])
            
    print(list4)
    print(len(list4))
    with open(filename3, "wb") as file:
        pickle.dump(list4, file, protocol=0)

import os
import re
import numpy as np
import torch
path1 = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/sent/'
sent = os.listdir(path1)

path2 = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/events_midput/'
event = os.listdir(path2)

for i in event:
    event_path = path2 + i
    file = open(event_path, "rb")
    # f = open(path,encoding = "utf-8")
    lines = file.readlines()
    for idx, line in enumerate(lines):
        size = len(line)
        sent_path = path1 + i
        sent_file = open(sent_path, "rb")
        # f = open(path,encoding = "utf-8")
        sent_lines = sent_file.readlines()
        sents = sent_lines[idx]
        if size == 2:
            s = str(sents)[2:-5]
            tmpt = s.replace(".", " ") \
                .replace(",", " ") \
                .replace("-", " ") \
                .replace("?", " ") \
                .replace(":", " ") \
                .replace("!", " ") \
                .replace(";", " ") \
                .replace("(", " ") \
                .replace(")", " ") \
                .replace("--", " ") \
                .replace("\"", " ") \
                .split()
            num = len(tmpt)

            b = np.ones(num,dtype=np.int)
            output = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/' + "04_events_sent/"
            if not os.path.exists(output):
                os.makedirs(output)
            filename = output + i
            with open(filename, "a", encoding="ascii") as file:
                file.write(str(b))
                file.write('\r\n')
                file.close()

        if size > 2:
            s = str(sents)[2:-5]
            s = s.replace("\\","")
            #s = re.sub('[\]','',s)
            output = '/prot/lkz/my_project/my_project-gist/text-gist/text_gist-results/' + "04_events_sent/"
            if not os.path.exists(output):
                os.makedirs(output)
            filename = output + i
            with open(filename, "a", encoding="ascii") as file:
                 file.write(s)
                 file.write('\r\n')
                 file.close()

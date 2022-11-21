import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

bold_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/bold_events/'
text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/text_events/text_events/'

for i in os.listdir(bold_path):
    files_path = bold_path + i + "/"
    story = os.listdir(files_path)
    story.sort()
    for j in story:
        k = j[0:-4]
        file_path = files_path + j
        data = np.load(file_path, allow_pickle=True)
        text_file = text_path + j
        datas = np.load(text_file)
        RSM = np.corrcoef(data)
        text_RSM = np.corrcoef(datas)
        output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/bold_events_RSM/' + i +"/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, k + "-RSM"), RSM)
        output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/text_events_RSM/' + k + "/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output, "text_" + k + "-RSM"), text_RSM)
        # Plot figure of these correlations
        '''f, ax = plt.subplots(1, 1, figsize=(8, 7))
        plt.imshow(
            RSM,
            cmap='bwr',
            vmin=-1,
            vmax=1,
        )
        plt.colorbar()
        ax.set_title('%s' % (k))
        ax.set_xlabel('events')
        ax.set_ylabel('events')
        out = '/protNew/lkz/my_project/my_project-gist/onebrain/rsa/each_story/result/bold_events_RSM_plot/' + i +"/"
        if not os.path.exists(out):
            os.makedirs(out)
        plt.savefig(out + k + "-RSM.png")
        plt.show()'''

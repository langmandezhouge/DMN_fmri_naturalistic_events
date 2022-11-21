import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from scipy.stats import spearmanr,pearsonr
from scipy import stats

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

bold_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/bold_events_RSM/'
text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/text_events_RSM/'

for i in os.listdir(bold_path):
    files_path = bold_path + i + "/"
    dataset = []
    for j in os.listdir(files_path):
        k = j[0:-8]
        bold_file = files_path + j
        bold_data = np.load(bold_file, allow_pickle=True)
        #bold_data = torch.tensor(bold_data)
        text_data = np.load(text_path + k + "/" + "text_" + k + "-RSM.npy", allow_pickle=True)
        #text_data = torch.tensor(text_data)
        bold = upper_tri_masking(bold_data)
        text = upper_tri_masking(text_data)

        #r1,p1 = pearsonr(bold,text)
        r,p = stats.spearmanr(bold, text,axis=None)
        output1 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_result/rsa_r/' + i + "/"
        if not os.path.exists(output1):
            os.makedirs(output1)
        np.save(os.path.join(output1, "rsa-" + k), r)
        output1 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/each_story/result/rsa_result/pearson-rsa_result/rsa_p/' + i + "/"
        if not os.path.exists(output1):
            os.makedirs(output1)
        np.save(os.path.join(output1, "rsa-" + k), p)
        # Plot figure of these correlations
        '''f, ax = plt.subplots(1, 1, figsize=(8, 7))
        plt.imshow(
            RSM,
            cmap='bwr',
            vmin=-1,
            vmax=1,
        )
        plt.colorbar()
        ax.set_title('%s-text-RSM' % (i + k))
        ax.set_xlabel('events')
        ax.set_ylabel('events')
        out = '/protNew/lkz/my_project/my_project-gist/onebrain/rsa/each_story/result/rsa_result/rsa_result_plot/' + i + "/"
        if not os.path.exists(out):
            os.makedirs(out)
        plt.savefig(out + "rsa-" + k + "-RSM.png")
        plt.show()'''

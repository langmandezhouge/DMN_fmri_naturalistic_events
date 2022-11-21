import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import os
import numpy as np
from scipy import stats
from scipy.stats import spearmanr,pearsonr

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

bold_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/bold_events_RSM/'
text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/text_events_RSM/text_events_RSM.npy'
#bold_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/bold_events_RSM/'
#text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/text_events_RSM/text_events_RSM.npy'
for i in os.listdir(bold_path):
    j = i[7:10]
    bold_file = bold_path + i
    bold_data = np.load(bold_file,allow_pickle=True)
    #bold_data = torch.tensor(bold_data)
    text_data = np.load(text_path,allow_pickle=True)
    bold = upper_tri_masking(bold_data)
    text = upper_tri_masking(text_data)

    r, p = stats.spearmanr(bold, text)
    output1 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/rsa_r/'
    #output1 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/rsa_result/rsa_result/rsa_r/'
    if not os.path.exists(output1):
        os.makedirs(output1)
    np.save(os.path.join(output1, "rsa-" + j), r)
    output2 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/rsa_p/'
    #output2 = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/rsa_result/rsa_result/rsa_p/'
    if not os.path.exists(output2):
        os.makedirs(output2)
    np.save(os.path.join(output2, "rsa-" + j), p)

    '''# Plot figure of these correlations
    f, ax = plt.subplots(1, 1, figsize=(8, 7))
    plt.imshow(
        r,
        cmap='bwr',
        vmin=-1,
        vmax=1,
    )
    plt.colorbar()
    ax.set_title('%s-text-RSM' % ('region-' + j))
    ax.set_xlabel('events')
    ax.set_ylabel('events')
    plt.savefig("/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result_plot/" + "rsa-" + j + "-RSM.png")
    plt.show()'''

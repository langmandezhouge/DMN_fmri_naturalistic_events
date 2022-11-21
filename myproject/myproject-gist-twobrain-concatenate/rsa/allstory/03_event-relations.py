import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import numpy as np
import matplotlib.pyplot as plt
import torch
import os



text_path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/text_events.npy'
text_data = np.load(text_path,allow_pickle=True)
text_dataset = torch.tensor(text_data)
text_RSM = np.corrcoef(text_dataset)
path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/bold_events/'
#path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/bold_events/'

for i in os.listdir(path):
    j = i[0:10]
    file_path = path + i
    data = np.load(file_path, allow_pickle=True)
    dataset = torch.tensor(data)
    RSM = np.corrcoef(dataset)

    output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/bold_events_RSM/'
    #output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/6story/result/bold_events_RSM/'
    if not os.path.exists(output):
        os.makedirs(output)
    np.save(os.path.join(output, j + "-RSM"), RSM)

output = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/text_events_RSM/'
if not os.path.exists(output):
    os.makedirs(output)
np.save(os.path.join(output, "text_events_RSM"), text_RSM)

# Plot figure of these correlations'''
'''f, ax = plt.subplots(1,1, figsize=(8, 7))
plt.imshow(
    RSM,
    cmap='bwr',
    vmin=-1,
    vmax=1,
)
plt.colorbar()
ax.set_title('%s' % ('text_events_RSM'))
ax.set_xlabel('events')
ax.set_ylabel('events')
plt.savefig("/protNew/lkz/my_project/my_project-gist/rsa/allstory/result/text_events_RSM/text_events_RSM.png")
plt.show()'''

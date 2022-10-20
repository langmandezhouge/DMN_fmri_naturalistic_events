import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import numpy as np
import os
import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn import plotting
from scipy import stats
from scipy.ndimage.measurements import center_of_mass
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

input_dir = '/prot/lkz/searchlight/Naturalistic/sub-001_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz'
nii = nib.load(input_dir)
print(nii.shape)

atlas_filename = '/prot/lkz/LSTM/results/mask/atlas_246.nii'
# Plot the ROIs
plotting.plot_roi(atlas_filename);
print('Harvard-Oxford cortical atlas')
plt.show()

# Create a masker object that we can use to select ROIs
masker_ho = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True)
#print(masker_ho.get_params())

# Apply our atlas to the Nifti object so we can pull out data from single parcels/ROIs
bold_ho = masker_ho.fit_transform(nii)
#print('shape: parcellated bold time courses: ', np.shape(bold_ho))

roi_id = 34
bold_ho_pPHG_r = np.array(bold_ho[:, roi_id])
bold_ho_pPHG_r = bold_ho_pPHG_r.reshape(bold_ho_pPHG_r.shape[0], -1)
print("Posterior PPC (region 35) rightward attention trials: ", bold_ho_pPHG_r.shape)

plt.figure(figsize=(14,4))
plt.plot(bold_ho_pPHG_r)
plt.ylabel('Evoked activity')
plt.xlabel('Timepoints')
sns.despine()
plt.show()

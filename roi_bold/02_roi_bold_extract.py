import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

import nibabel as nib
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
from nilearn import plotting
import matplotlib.pyplot as plt
from nilearn import input_data


input_dir = '/prot/lkz/searchlight/Naturalistic/sub-001_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz'
nii = nib.load(input_dir)
print(nii.shape)

atlas_filename = '/prot/lkz/LSTM/results/mask/roi_003.nii.gz'
# Plot the ROIs
plotting.plot_roi(atlas_filename);
print('Harvard-Oxford cortical atlas')
plt.show()

# Init a masker object that also standardizes the data
masker_wb = input_data.NiftiMasker(mask_img=atlas_filename,standardize=True)

# Pull out the time course for voxel
bold_wb = masker_wb.fit_transform(nii)

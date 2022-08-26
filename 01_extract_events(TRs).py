#提取事件（指定时间点TR），并保存为nii格式

from nilearn.image import load_img
import nibabel as nib
import nibabel
import numpy as np
import os
from nilearn import masking
from pathlib import Path

input_dir = '/prot/lkz/searchlight/Naturalistic/sub-001_task-pieman_run-1_space-MNI152NLin2009cAsym_res-native_desc-sm6_bold.nii.gz'
output_dir = '/prot/lkz/searchlight/Naturalistic/results/'
output_name = 'data.nii.gz'

fMRIData = nib.load(input_dir)
nii = fMRIData.get_data()

# extract data for this subject,the fourth dimension is TRs(timepoints)
data_nii = nii[:,:,:,0:15]
#print(data_nii)

# View non-zero values in the matrix
print(np.nonzero(data_nii))

# extract affine
affine_mat = fMRIData.affine  # What is the orientation of the data
dimsize = (3.0, 3.0, 4.0, 1.5) #TRs=1.5

## save numpy to nii file
bold_nii = nib.Nifti1Image(data_nii,affine_mat)
hdr = bold_nii.header  # get a handle for the .nii file's header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
nib.save(bold_nii, os.path.join(output_dir, output_name))

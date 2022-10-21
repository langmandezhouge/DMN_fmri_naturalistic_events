#make each roi and save
import numpy as np
import nibabel as nib
import os
from nilearn.input_data import NiftiLabelsMasker

basePath = '/prot/lkz/brain_roi_mask/'
atlasName = "atlas_246.nii"
atlasFile = os.path.join(basePath, atlasName)

atlas_nii = nib.load(atlasFile)
atlas_arr = atlas_nii.get_fdata()

mask_arr = atlas_arr.copy()
for i in range(246):
    roiIndex = i +1
    mask_arr[atlas_arr != roiIndex] = 0
    mask_arr[atlas_arr == roiIndex] = 1
    mask_affine = atlas_nii.affine.copy()
    mask_hrd = atlas_nii.header.copy()
    mask_hrd["cal_max"] = 1
    # print(mask_hrd)
    output = '/prot/lkz/LSTM/roi_results/roi_mask/'
    mask_nii = nib.Nifti1Image(mask_arr, mask_affine, mask_hrd)
    nib.save(mask_nii, os.path.join(output, "roi_" + str('%03d'%roiIndex) + ".nii.gz"))

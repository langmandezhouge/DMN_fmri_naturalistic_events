import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.plotting import view_img, view_img_on_surf
from nltools.mask import expand_mask, roi_to_brain
import os
from nilearn import plotting
from nibabel.viewers import OrthoSlicer3D

path = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/rsa_result/p_map_allstory/p_map_allstory.npy'
file = np.load(path)
mask_path = '/protNew/lkz/my_project/roi_mask/atlas_246.nii'
mask = nib.load(mask_path)

data = roi_to_brain(pd.Series(file), expand_mask(mask))
result = view_img(data.to_nifti(),cmap='rainbow')
#surf_result = view_img_on_surf(data.to_nifti(),threshold=0.05, cmap='jet')
out = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/map_brain/allstory/p_0.05/'
#out = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/map_brain/allstory/p_surf/'
if not os.path.exists(out):
    os.makedirs(out)
result.save_as_html(out + 'p_allstory.html')
#surf_result.save_as_html(out + 'p_allstory.html')

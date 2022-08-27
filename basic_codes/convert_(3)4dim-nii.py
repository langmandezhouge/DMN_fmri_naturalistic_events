import numpy as np
import os
import nibabel as nib

# extract affine
input_dir = '/prot/lkz/searchlight/Naturalistic/results/data.nii.gz'
fMRIData = nib.load(input_dir)
affine_mat = fMRIData.affine  # What is the orientation of the data
dimsize = (3.0, 3.0, 4.0, 1.5)

inputdim_dir = 'x.npy'
data = np.load(inputdim_dir)
#data = 4dim.numpy #3/4dim numpy
data = data.astype('double')  # Convert the output into a precision format that can be used by other applications
data[np.isnan(data)] = 0  # Exchange nans with zero to ensure compatibility with other applications
data_nii = data

## save data_nii(numpy) to nii file
bold_nii = nib.Nifti1Image(data_nii,affine_mat)
hdr = bold_nii.header  # get a handle for the .nii file's header
hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
output_dir = '/'
output_name = 'y.nii.gz'
nib.save(bold_nii, os.path.join(output_dir, output_name))

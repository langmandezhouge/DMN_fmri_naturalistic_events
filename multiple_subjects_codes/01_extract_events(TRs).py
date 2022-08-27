'''extract events(specific TRs), and save to nii format file'''

from nilearn.image import load_img
import nibabel as nib
import nibabel
import numpy as np
import os
from nilearn import masking
from pathlib import Path


import os, sys
import numpy as np

path = '/prot/lkz/DMN_fmri_naturalistic_events/subjects/'

files = os.listdir(path)
files.sort(key=lambda x:int(x[4:]))
print(files)

count = len(files)
print(count)

for i in range(count):
    subj_path = path + 'sub-' + '%.2d' % (i+1) + '/' + '%.2d' % (i+1) + '.nii.gz'
    data = subj_path
    fMRIData = nib.load(data)
  #  print(data)
    print(fMRIData.shape)
    nii = fMRIData.get_data()

    # extract data for this subject,the fourth dimension is TRs(timepoints)
    data_nii = nii[:,:,:,200:250]
    #print(data_nii)

    # View non-zero values in the matrix
    print(np.nonzero(data_nii))

    # extract affine
    affine_mat = fMRIData.affine  # What is the orientation of the data
    dimsize = (3.0, 3.0, 4.0, 1.5) #TRs=1.5

    # save numpy to nii file
    bold_nii = nib.Nifti1Image(data_nii,affine_mat)
    hdr = bold_nii.header  # get a handle for the .nii file's header
    hdr.set_zooms((dimsize[0], dimsize[1], dimsize[2], dimsize[3]))
    
    output_dir = path + 'sub-' + '%.2d' % (i+1) + '/' + 'results' + '01_fmri_nii_events'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_dir + '/' + 'E' + '%.2d' % (i+1) + '.nii.gz',bold.nii)   

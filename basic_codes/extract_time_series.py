import nibabel as nib
import numpy as np
import os

input_dir = '/prot/lkz/searchlight/Naturalistic/results/data.nii.gz'
data = nib.load(inputdim_dir)

'''# way 1:
# Gain mask
from nilearn import masking
mask = masking.compute_background_mask(data)
# Gain the TimeSeries
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
masker = NiftiMasker(mask_img=mask, standardize=True)
time_series = masker.fit_transform(data)'''

# way 2:
from nilearn import masking
mask = masking.compute_background_mask(data)
print(mask.get_data().shape)
from nilearn.masking import apply_mask
time_series = apply_mask(data, mask)

print(time_series.shape)
#print(time_series)
#save the time series(numpy)
np.save(os.path.join(output_dir, 'time_series'),time_series)

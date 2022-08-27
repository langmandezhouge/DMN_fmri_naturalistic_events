import nibabel as nib
import numpy as np
import os

data = 

'''# way 1:
# Gain mask
from nilearn import masking
mask = masking.compute_background_mask(data_i)
# Gain the TimeSeries
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
masker = NiftiMasker(mask_img=mask, standardize=True)
time_series = masker.fit_transform(data_i)'''

# way 2:
from nilearn import masking
mask = masking.compute_background_mask(data_i)
print(mask.get_data().shape)
from nilearn.masking import apply_mask
time_series_02 = apply_mask(data_i, mask)

print(time_series_02.shape)
#print(time_series)
#save the time series(numpy)
np.save(os.path.join(output_dir, 'time_series_02'),time_series_02)

# -*- coding: utf-8 -*-
from nilearn import masking

data = '/prot/lkz/searchlight/Naturalistic/results/data.nii.gz'

mask = masking.compute_background_mask(data)

# Time-series from a brain parcellation
# Download atlas from internet
from nilearn import datasets
dataset = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = dataset.maps
labels = dataset.labels

# Apply atlas to my data
from nilearn.image import resample_to_img
Atlas = resample_to_img(atlas_filename, mask, interpolation='nearest')

# Gain the TimeSeries
from nilearn.input_data import NiftiLabelsMasker
masker = NiftiLabelsMasker(labels_img=Atlas, standardize=True,
                           memory='nilearn_cache', verbose=5)
dataset_2d = masker.fit_transform(data)

print(dataset_2d)
print(dataset_2d.shape)

# And now plot a few of these
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 5))
plt.plot(dataset_2d[:200, 998:1000]) #extract specified voxel features
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.xlim(0, 200) # TRs
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)

plt.show()

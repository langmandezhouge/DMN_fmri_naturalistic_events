'''Compute ISC'''

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os
import glob
import time
from copy import deepcopy
import numpy as np
import pandas as pd

from nilearn import datasets
from nilearn import surface
from nilearn import plotting
from nilearn.input_data import NiftiMasker, NiftiLabelsMasker
import nibabel as nib

from brainiak import image, io
from brainiak.isc import isc, isfc, permutation_isc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white', context='talk', font_scale=1, rc={"lines.linewidth": 2})


# way 1
path = '/prot/lkz/Pieman2_isc/Pieman2/'

dir_mask = os.path.join(path, 'masks/')
mask_name = os.path.join(dir_mask, 'avg152T1_gray_3mm.nii.gz')
all_task_names = ['word', 'intact1']

"""load brain template"""
# Load the brain mask
brain_mask = io.load_boolean_mask(mask_name)
# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)
# Load the brain nii image
brain_nii = nib.load(mask_name)

# Custom mean estimator with Fisher z transformation for correlations
def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):
    return np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))

task_isc = np.load('/prot/lkz/Pieman2_isc/Pieman2/results/isc/isc_maps_word.npy')

# Fisher Z-transformed mean across subjects
task_mean = compute_summary_statistic(task_isc, axis=0) # mean isc
print(task_mean.shape)


# way 2
# run ISC, loop over conditions
#def isc(data, pairwise=False, summary_statistic=mean, tolerate_nans=True)

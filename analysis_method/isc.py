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

path = '/prot/lkz/Pieman2_isc/Pieman2/'

dir_mask = os.path.join(path, 'masks/')
mask_name = os.path.join(dir_mask, 'avg152T1_gray_3mm.nii.gz')
all_task_names = ['word', 'intact1']
all_task_des = ['word level scramble', 'intact story']
n_subjs_total = 18
group_assignment_dict = {task_name: i for i, task_name in enumerate(all_task_names)}

results = '/prot/lkz/Pieman2_isc/Pieman2/results/'
# Where do you want to store the data
dir_out = results + 'isc/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
    print('Dir %s created ' % dir_out)


def get_file_names(data_dir_, task_name_, verbose=False):
    """
    Get all the participant file names

    Parameters
    ----------
    data_dir_ [str]: the data root dir
    task_name_ [str]: the name of the task

    Return
    ----------
    fnames_ [list]: file names for all subjs
    """
    c_ = 0
    fnames_ = []
    # Collect all file names
    for subj in range(1, n_subjs_total+1):
        fname = os.path.join(
            data_dir_, 'sub-%.3d/func/sub-%.3d-task-%s.nii.gz' % (subj, subj, task_name_))

        # If the file exists
        if os.path.exists(fname):

            # Add to the list of file names
            fnames_.append(fname)

    return fnames_


"""load brain template"""
# Load the brain mask
brain_mask = io.load_boolean_mask(mask_name)

# Get the list of nonzero voxel coordinates
coords = np.where(brain_mask)

# Load the brain nii image
brain_nii = nib.load(mask_name)


"""load bold data"""
# load the functional data
fnames = {}
images = {}
masked_images = {}
bold = {}
group_assignment = []
n_subjs = {}

for task_name in all_task_names:
    data_path = '/prot/lkz/Pieman2_isc/Pieman2/'
    fnames[task_name] = get_file_names(data_path, task_name)
    images[task_name] = io.load_images(fnames[task_name])
    masked_images[task_name] = image.mask_images(images[task_name], brain_mask)
    # Concatenate all of the masked images across participants
    bold[task_name] = image.MaskedMultiSubjectData.from_masked_images(
        masked_images[task_name], len(fnames[task_name])
    )
    # Convert nans into zeros
    bold[task_name][np.isnan(bold[task_name])] = 0
    # compute the group assignment label
    n_subjs_this_task = np.shape(bold[task_name])[-1]
    group_assignment += list(
        np.repeat(group_assignment_dict[task_name], n_subjs_this_task)
    )
    n_subjs[task_name] = np.shape(bold[task_name])[-1]
    print('Data loaded: {} \t shape: {}' .format(task_name, np.shape(bold[task_name])))


'''Compute ISC'''
# run ISC, loop over conditions
isc_maps = {}
for task_name in all_task_names:
    isc_maps[task_name] = isc(bold[task_name], pairwise=False)
    # isc_maps[task_name] = isc(bold[task_name], pairwise=False, summary_statistic='mean') # mean isc
    np.save(os.path.join(dir_out, 'isc_maps_%s' % (task_name)), isc_maps[task_name])
    print('Shape of %s condition:' % task_name, np.shape(isc_maps[task_name]))
    subj_num = np.shape(isc_maps[task_name])[0]
    for subj_id in range(0,subj_num):

        # Make the ISC output a volume
        isc_vol = np.zeros(brain_nii.shape)
        # Map the ISC data for the first participant into brain space
        isc_vol[coords] = isc_maps[task_name][subj_id, :]
        # make a nii image of the isc map
        isc_nifti = nib.Nifti1Image(isc_vol, brain_nii.affine, brain_nii.header)

        # Save the ISC data as a volume
        subj_dir = dir_out + 'sub-%.3d'%(subj_id+1) +'/'
        if not os.path.exists(subj_dir):
            os.makedirs(subj_dir)
        isc_map_path = os.path.join(subj_dir, 'ISC_%s_sub-%.3d.nii.gz' % (task_name, subj_id+1))
        nib.save(isc_nifti, isc_map_path)

        # Plot the data as a statmap
        threshold = .2

        f, ax = plt.subplots(1, 1, figsize=(12, 5))
        plotting.plot_stat_map(
            isc_nifti,
            threshold=threshold,
            axes=ax
        )
        ax.set_title('ISC map for subject {}, task = {}'.format(subj_id+1, task_name))
        plt.savefig(os.path.join(subj_dir + 'ISC_%s_sub-%.3d.png' % (task_name, subj_id+1)),bbox_inches='tight', dpi=450)
        plt.show()
        
        
        
        '''volumetric space to surface space'''

        view = 'medial'

        # get a surface
        fsaverage = datasets.fetch_surf_fsaverage()

        # make "texture"
        texture = surface.vol_to_surf(isc_nifti, fsaverage.pial_left)

        # plot
        title_text = ('Left Avg ISC map, {} for sub-%.3d'.format(task_name) % (subj_id + 1))
        surf_left_map = plotting.plot_surf_stat_map(
            fsaverage.infl_left, texture,
            hemi='left', view=view,
            title=title_text,
            threshold=threshold, cmap='RdYlBu_r',
            colorbar=True,
            bg_map=fsaverage.sulc_left)
        plt.savefig(os.path.join(subj_dir + 'fsaverage_left_ISC_%s_sub-%.3d.png' % (task_name, subj_id + 1)),
                    bbox_inches='tight', dpi=450)
        plt.show()

        # plot
        title_text = ('Right Avg ISC map, {} for sub-%.3d'.format(task_name) % (subj_id + 1))
        surf_right_map = plotting.plot_surf_stat_map(
            fsaverage.infl_right, texture,
            hemi='right', view=view,
            title=title_text,
            threshold=threshold, cmap='RdYlBu_r',
            colorbar=True,
            bg_map=fsaverage.sulc_right)

        plt.savefig(os.path.join(subj_dir + 'fsaverage_right_ISC_%s_sub-%.3d.png' % (task_name, subj_id + 1)),
                    bbox_inches='tight', dpi=450)
        plt.show()

# -*- coding: utf-8 -*-

import warnings
import sys

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Import libraries
import nibabel as nib
import numpy as np
import os
import time
from nilearn import plotting
from brainiak.searchlight.searchlight import Searchlight
from brainiak.fcma.preprocessing import prepare_searchlight_mvpa_data
from brainiak import io
import pandas as pd
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import masking

path = '/prot/lkz/DMN_fmri_naturalistic_events/results/01_fmri_nii_events/'
#brain_mask = '/prot/lkz/searchlight/Naturalistic/tpl-MNI152NLin2009cAsym_res-pieman_desc-brain_mask.nii.gz' 

files = os.listdir(path)
files.sort(key=lambda x:int(x[1:-7]))
print(files)

count = len(files)
print(count)

for i in range(count):
    bold_vol = files[i]
    bold_vol = nib.load(path + bold_vol)
    print(bold_vol.shape)
    brain_mask = masking.compute_background_mask(bold_vol)
    print(brain_mask.get_data().shape) 
     
    affine_mat = bold_vol.affine
    dimsize = (3.0, 3.0, 4.0, 1.5)

    whole_brain_mask = brain_mask.get_data() 
    # Preset the variables to be used in the searchlight
    bold = bold_vol.get_data()
    data = bold
    mask = whole_brain_mask
    bcvar = None 
    sl_rad = 1
    max_blk_edge = 5
    pool_size = 1


    # Start the clock to time searchlight
    begin_time = time.time()

    # Create the searchlight object
    sl = Searchlight(sl_rad=sl_rad,max_blk_edge=max_blk_edge)
    print("Setup searchlight inputs")
    print("Input data shape: " + str(data.shape))
    print("Input mask shape: " + str(mask.shape) + "\n")

    # Distribute the information to the searchlights (preparing it to run)
    sl.distribute([data], mask)

    # Data that is needed for all searchlights is sent to all cores via the sl.broadcast function.
    sl.broadcast(bcvar)

    output_dir = '/prot/lkz/DMN_fmri_naturalistic_events/results/02_searchlight_events-matrix/'

    df = pd.DataFrame()

    # Set up the kernel
    def test(dataset, mask, mysl_rad, bcvar):
        dataset = np.array(dataset)
        print(dataset.shape)
        datas = np.reshape(dataset, ( 3 * 3 * 3,15)) 
        print(datas.shape)
        datas = np.mean(datas, axis=1) 
        print(datas.shape)
        import pandas as pd
        datas = pd.DataFrame(datas) 
        datas = np.transpose(datas) 
        # print(datas)
        global df
        df = df.append(datas, ignore_index=True)
        print(df.shape)
        print(df)


    print("Begin Searchlight\n")
    sl_result = sl.run_searchlight(test, pool_size=pool_size) 

    np.save(os.path.join(output_dir,  'searchlight_E' + str(i)), df)
    #np.save(os.path.join(output_dir, 'sl_result'),sl_result)
    #print(sl_result.shape)
    print("End Searchlight\n")

    end_time = time.time()

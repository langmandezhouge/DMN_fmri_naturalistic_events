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
from brainiak import io
import pandas as pd
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import masking

path = '/prot/lkz/LSTM/data.nii.gz'

count = 1

for i in range(count):
    bold_vol = nib.load(path)
    print(bold_vol.shape)
    brain_mask = masking.compute_background_mask(bold_vol)
    print(brain_mask.get_data().shape)

    #    bold_vol = nib.load(path + bold_vol)
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
    sl = Searchlight(sl_rad=sl_rad, max_blk_edge=max_blk_edge)
    print("Setup searchlight inputs")
    print("Input data shape: " + str(data.shape))
    print("Input mask shape: " + str(mask.shape) + "\n")

    # Distribute the information to the searchlights (preparing it to run)
    sl.distribute([data], mask)

    # Data that is needed for all searchlights is sent to all cores via the sl.broadcast function.
    sl.broadcast(bcvar)

    output_dir = '/prot/lkz/LSTM/results/'

    df = pd.DataFrame()


    # Set up the kernel
    def test(dataset, mask, mysl_rad, bcvar):
        dataset = np.array(dataset)
        print(dataset.shape)
        a = dataset.shape[4]
        b = dataset.shape[1]
        datas = np.reshape(dataset, (b * b * b, a))
        print(datas.shape)
        import pandas as pd
        datas = pd.DataFrame(datas)
        # print(datas)
        global df
        df = df.append(datas, ignore_index=True)
        print(df.shape)
        print(df)


    print("Begin Searchlight\n")
    sl_result = sl.run_searchlight(test, pool_size=pool_size)

    np.save(os.path.join(output_dir, 'searchlight_E'), df)
    # np.save(os.path.join(output_dir, 'sl_result'),sl_result)
    # print(sl_result.shape)
    print("End Searchlight\n")

    end_time = time.time()

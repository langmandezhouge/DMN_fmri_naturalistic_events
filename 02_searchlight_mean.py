#将已提取的事件nii文件用searchlight方法对每一个体素提取信号（1个体素的数值用27个体素值来表示）
#每个体素进行searchlight后按时间点进行了平均，得到（27，1）的向量，并转置为（1，27）向量；所有体素的1维向量连接保存为一个文件
#结果最终保存为一个事件的所有体素对应的seachlight数据（n,27),n为体素总数（52422）

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

input_dir = '/prot/lkz/searchlihgt_pearson/pearson/data.nii.gz'
#brain_mask = '/prot/lkz/searchlight/Naturalistic/tpl-MNI152NLin2009cAsym_res-pieman_desc-brain_mask.nii.gz' #加载brain_mask，也可以自动生成
from nilearn import masking
brain_mask = masking.compute_background_mask(input_dir) #根据nii文件自动生成brain_mask
print(brain_mask.get_data().shape)  #mask中有很多信息，获取mask中数据大小

bold_vol = nib.load(input_dir)
affine_mat = bold_vol.affine
dimsize = (3.0, 3.0, 4.0, 1.5)

whole_brain_mask = brain_mask.get_data() #mask中有很多信息，获取mask中数据

# Preset the variables to be used in the searchlight
bold = bold_vol.get_data()
data = bold
mask = whole_brain_mask
bcvar = None #没有标签时，设置为None
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

output_dir = '/prot/lkz/searchlihgt_pearson/pearson/results/'

df = pd.DataFrame()

# Set up the kernel
def test(dataset, mask, mysl_rad, bcvar):
    dataset = np.array(dataset)
    print(dataset.shape)
    datas = np.reshape(dataset, ( 3 * 3 * 3,15)) #将多维数组（1，3，3，3，15）拉成2维数组（27，15）
    print(datas.shape)
    datas = np.mean(datas, axis=1) #将2维度数组按列平均，得到一维向量（27，1）
    print(datas.shape)
    import pandas as pd
    datas = pd.DataFrame(datas) #将数据转化为标准的列表格式
    datas = np.transpose(datas) # 转置函数，将（27，1）转置为（1，27）
   # print(datas)
    global df
    df = df.append(datas, ignore_index=True) #将后面生成的数据按行保存在前一个数据的后面
    print(df.shape)
    print(df)


print("Begin Searchlight\n")
sl_result = sl.run_searchlight(test, pool_size=pool_size) #运行searchlight函数

np.save(os.path.join(output_dir, 'E1'), df) #将searchlig得到的数据df保存为事件E1（numpy格式）

np.save(os.path.join(output_dir, 'sl_result'),sl_result)
#print(sl_result.shape)
print("End Searchlight\n")

end_time = time.time()

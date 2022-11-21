import numpy as np
import os

path = '/protNew/lkz/my_project/Narrative-result/allbold_allsub/'
region_path = os.listdir(path)
region_path.sort()

for i in region_path:
    r = i[-3:]
    m = int(r) - 1
    if int(r)%2 == 0:
        region_file = path + i + "/"
        region = os.listdir(region_file)
        region.sort()
        for j in region:
            story_file = region_file + j + "/"
            story = os.listdir(story_file)
            story.sort()
            for k in story:
                n = k[-16:-9]
                data1_path = story_file + k
                data1_file = np.load(data1_path, allow_pickle=True)
                data2_path = path + "region-" + '%03d'%(m) + "/" + j + "/" + "roi-" + '%03d'%(m) + "_" + j + "_" + n + "_bold.npy"
                data2_file = np.load(data2_path, allow_pickle=True)
                data = np.append(data1_file,data2_file,axis=0)

                output = '/protNew/lkz/my_project/Narrative-result/twobrain_allbold_allsub/' + i + "/" + j + "/"
                if not os.path.exists(output):
                    os.makedirs(output)
                np.save(os.path.join(output, k), data)

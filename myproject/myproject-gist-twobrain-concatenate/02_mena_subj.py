import numpy as np
import os

path = '/protNew/lkz/my_project/Narrative-result/twobrain_allbold_allsub/'
for i in os.listdir(path):
    r = i[-3:]
    region_file = path + i + "/"
    region = os.listdir(region_file)
    region.sort()
    for j in region:
        story_file = region_file + j + "/"
        data = 0
        story = os.listdir(story_file)
        story.sort()
        for k in story:
            data_path = story_file + k
            data_file = np.load(data_path,allow_pickle=True)
            data += data_file
        num = len(os.listdir(story_file))
        data_mean = data/num
        output = '/protNew/lkz/my_project/Narrative-result/' + "twobrain_allbold_sub_mean/" + i + "/" + j + "/"
        if not os.path.exists(output):
            os.makedirs(output)
        np.save(os.path.join(output,  "roi-" + r + "_" + j + "_mean_bold.npy"), data_mean)

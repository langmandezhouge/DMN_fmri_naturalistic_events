import numpy as np
import os
arr = np.random.randint(0,2,246)

data = []
for i,row in enumerate(arr):
    if i ==214:
       row = 1
    else:
        row = 0
    data.append(row)

out = '/protNew/lkz/my_project/my_project-gist/twobrain-concatenate/rsa/allstory/result/rsa_result/pearson-rsa_story/' + "p_map_allstory/"
if not os.path.exists(out):
    os.makedirs(out)
np.save(os.path.join(out, "p_map_a"), data)
'''np.save(os.path.join(out, "p_map_forgot"), data)
np.save(os.path.join(out, "p_map_lucy"), data)
np.save(os.path.join(out, "p_map_merlin"), data)
np.save(os.path.join(out, "p_map_notthefallintact"), data)
np.save(os.path.join(out, "p_map_original"), data)
np.save(os.path.join(out, "p_map_synonyms"), data)
np.save(os.path.join(out, "p_map_vodka"), data)
np.save(os.path.join(out, "p_map_shapesphysical"), data)
np.save(os.path.join(out, "p_map_tunnel"), data)
np.save(os.path.join(out, "p_map_piemanpni"), data)'''

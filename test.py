
import os
import random
import numpy as np

base_path = "static/img/"
images_folder = "profile/nethika"

prof_path = base_path + images_folder

im_list = os.listdir(prof_path)
# choose random 3
ims = np.random.choice(im_list, 3)

print (ims)

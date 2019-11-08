import os
import matplotlib.pyplot as plt
import numpy as np

import nibabel as nib
from nibabel.testing import data_path
from nilearn import image
from nilearn import plotting


img_path = "/home/evan/Downloads/ds000221_R1.0.0/sub-010004/ses-02/func/sub-010004_ses-02_task-rest_acq-AP_run-01_bold.nii.gz"
# img_path = os.path.join(data_path, 'example4d.nii.gz')
# img = nib.load(img_path)
# # print(img)
# print(img.shape)

# plotting.plot_img(example_filename)

# img3d = image.index_img(img_path, 0)
# print(img3d.shape)
# plotting.plot_stat_map(img3d)
# plotting.plot_img(img3d)
# plt.show()

print("bigboi")
all_images = list(image.iter_img(img_path))
for img in all_images[:5]:
    # img is now an in-memory 3D img
    plotting.plot_stat_map(img, threshold=3, display_mode="z", cut_coords=1, colorbar=False)
plotting.show()

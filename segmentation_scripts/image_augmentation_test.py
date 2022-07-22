from image_augmentation import image_augmentation
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

'''
this file randomly chose one image to test the result of image aumentation class
namely training_scan/CT/001_Abdomen__2@0__I30f__3_win.nii
'''
img_arr = sitk.GetArrayFromImage(sitk.ReadImage(
    'X:/dataset/recurrence/training_scan/CT/001_Abdomen__2@0__I30f__3_win.nii')).transpose([1,2,0])
label_arr = sitk.GetArrayFromImage(sitk.ReadImage(
    'X:/dataset/recurrence/training_scan/livermask/001_Abdomen__2@0__I30f__3_win.nii')).transpose([1,2,0])



values, bin_edge = np.histogram(img_arr.ravel(),bins=256)
figure, axes = plt.subplots(1,2)
obj = image_augmentation(img_arr, label_arr,['rotate','contrast','gaussianNoise','GaussianBlur'])
augd_img, augd_label = obj.Execute()
# np.set_printoptions(threshold=512*512)
# print(augd_img[:,:,50])
ax_img, ax_hist = obj.display_img_hist(augd_img,axes=axes)
ax_img.set_title('image')
plt.show()

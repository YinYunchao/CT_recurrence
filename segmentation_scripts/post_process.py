import os
import numpy as np
import SimpleITK as sitk
from skimage import measure


ct_path = 'C:/Users/yyc13/recurrence/data/predict'
ct_list = sorted(os.listdir(ct_path))


ct_controlGroup_path = 'D:/recurrence_ControlGroup(nifti)/liver_mask(batch3)'
ct_controlGroup_list = sorted(os.listdir(ct_controlGroup_path))

ind_i = 0
for ct_scan in ct_controlGroup_list:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(ct_controlGroup_path,ct_scan))).transpose([1,2,0])
    labeled_img = measure.label(img, background=0)
    print(ind_i, ct_scan,'--',np.max(labeled_img))
    max_num=0
    for i in range(1,np.max(labeled_img)+1):
        if np.sum(labeled_img==i) > max_num:
            max_num = np.sum(labeled_img==i)
            max_pixel = i
        # print(max_num,':',max_pixel,'--',i)
    labeled_img[labeled_img != max_pixel] = 0
    labeled_img[labeled_img == max_pixel] = 1
    labeled_img = labeled_img.astype(np.int8)
    processed_itk = sitk.GetImageFromArray(labeled_img.transpose([2,0,1]))
    sitk.WriteImage(processed_itk,os.path.join('D:/recurrence_ControlGroup(nifti)/postprocess_liver_mask',ct_scan))
    # print(ind_i,ct_scan)
    ind_i+=1

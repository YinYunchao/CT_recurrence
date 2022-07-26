import os
import random 
import SimpleITK as sitk
import torch
import numpy as np
import matplotlib.pyplot as plt
from image_augmentation import image_augmentation

class ImageLoadPipe(object):
    """
    This is a standard pytorch image loader pipeline, check documentation on pytorch website
    take para of a list of nifti names and the path to it
    """
    def __init__(self, scan_path, scan_list, mask_path, slice_num = 32, if_transform=None, augmentation_list = None):
        self.scan_path = scan_path
        self.mask_path = mask_path
        self.scan_list = scan_list
        self.if_transform = if_transform
        self.transform = image_augmentation
        self.augmentation_list = augmentation_list
        self.slice_num =slice_num


    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self,idx):

        img_arr, mask_arr = self.read_nii(os.path.join(self.scan_path, self.scan_list[idx]),
                                          os.path.join(self.mask_path, self.scan_list[idx]),
                                          slice_num=self.slice_num)
        if self.if_transform:
            obj = self.transform(img_arr, mask_arr, self.augmentation_list)
            img_arr, mask_arr = obj.Execute()
        img_arr = np.expand_dims(img_arr, axis = 0)
        mask_arr = np.expand_dims(mask_arr, axis = 0)
        img_tensor = torch.from_numpy(img_arr)
        mask_tensor = torch.from_numpy(mask_arr)
        return img_tensor, mask_tensor


    def read_nii(self, img_path, mask_path, slice_num):
        """
        This function read image from nifti file and return it as torch tensor
        """
        img_itk = sitk.ReadImage(img_path)
        img_itk = self.resample(img_itk,
                                [img_itk.GetSpacing()[0]*2,img_itk.GetSpacing()[1]*2,img_itk.GetSpacing()[2]],
                                interpolator=sitk.sitkLinear)
        img_arr = sitk.GetArrayFromImage(img_itk).transpose([1,2,0])
        img_arr = img_arr.astype(np.float32)

        mask_itk = sitk.ReadImage(mask_path)
        mask_itk = self.resample(mask_itk,
                                [mask_itk.GetSpacing()[0]*2,mask_itk.GetSpacing()[1]*2,mask_itk.GetSpacing()[2]],
                                interpolator=sitk.sitkNearestNeighbor)
        mask_arr = sitk.GetArrayFromImage(mask_itk).transpose([1,2,0])
        mask_arr[mask_arr[:, :, :] > 0] = 1.0
        mask_arr = mask_arr.astype(np.float32)
        slice_ind = random.randint(0,img_arr.shape[2]-slice_num)
        return img_arr[:,:,slice_ind:slice_ind+slice_num], mask_arr[:,:,slice_ind:slice_ind+slice_num]

    def resample(self, img_itk, new_spacing, interpolator = sitk.sitkNearestNeighbor):
        '''
        a func to resample the image and change the resolution
        '''
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetInterpolator(interpolator)
        resample_filter.SetOutputDirection(img_itk.GetDirection())
        resample_filter.SetOutputOrigin(img_itk.GetOrigin())
        resample_filter.SetOutputSpacing(new_spacing)
        new_size = np.ceil(np.array(img_itk.GetSize()) * np.array(img_itk.GetSpacing()) / new_spacing)
        resample_filter.SetSize([int(new_size[0]), int(new_size[1]), int(new_size[2])])
        resampled_img = resample_filter.Execute(img_itk)
        return resampled_img


    def img_display(self, img, mask):
        '''
        this func display the img and mask, the middle slice as well
        '''
        figure = plt.figure(figsize = (1,2))
        img_sliceid = int(img.shape[-1]/2)
        figure.add_subplot(1,2,1)
        plt.title('scan')
        plt.axis('off')
        plt.imshow(img[0, 0, :, :, img_sliceid], cmap = 'gray')
        figure.add_subplot(1, 2, 2)
        plt.title('mask')
        plt.axis('off')
        plt.imshow(mask[0, 0, :, :, img_sliceid], cmap='gray')
        plt.show()
    






import os
import pandas as pd
import SimpleITK as sitk
import torch

class ImageLoadPipe(object):
    """
    This is a standard pytorch image loader pipeline, check documentation on pytorch website
    """
    def __init__(self, tumor_csv,img_dir, transform=None, target_transform=None):
        self.tumor_csv = pd.read_excel(tumor_csv)
        self.patient_id = self.tumor_csv['ID'].values
        self.seg_id = self.tumor_csv['Seg_id'].values
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform  = target_transform

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.img_name_gen(idx))
        img_tensor = self.read_nii(img_path)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        if self.target_transform:
            img_tensor = self.transform(img_tensor)
        return img_tensor


    def img_name_gen(self,idx):
        """
        this series of function generates the name of CT, SUV map and lable name
        """
        return str('{}_CT_{}.nii'.format(self.patient_id[idx],self.seg_id[idx].split('_')[-1]))

    def label_name_gen(self,idx):
        return str('{}_VOI_{}.nii'.format(self.patient_id[idx],self.seg_id[idx].split('_')[-1]))

    def suv_name_gen(self,idx):
        return str('{}_SUV_{}.nii'.format(self.patient_id[idx],self.seg_id[idx].split('_')[-1]))

    def read_nii(self, path):
        """
        This function read image from nifti file and return it as torch tensor
        """
        img_itk = sitk.ReadImage(path)
        img_arr = sitk.GetArrayFromImage(img_itk).transpose([1,2,0])
        img_tensor = torch.from_numpy(img_arr)
        return img_tensor



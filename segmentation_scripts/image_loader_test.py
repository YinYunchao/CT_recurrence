import image_loader
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
"""
This script tested image_loader 
should return a image tensor including images of one patient
"""
scan_path = 'X:/dataset/recurrence/training_scan/CT'
livermask_path = 'X:/dataset/recurrence/training_scan/livermask'
scan_list = os.listdir(scan_path)

liver_dataloader = image_loader.ImageLoadPipe(scan_path = scan_path,
                                        scan_list = scan_list,
                                        mask_path = livermask_path,
                                        if_transform = False,
                                        augmentation_list=['rotate','contrast','gaussianNoise','GaussianBlur'])
dlbcl_loader = DataLoader(liver_dataloader,batch_size = 1, shuffle = True)
print(len(dlbcl_loader))
img, mask = next(iter(dlbcl_loader))
print(img.shape)
print(mask.shape) 
liver_dataloader.img_display(img,mask)

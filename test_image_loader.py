import image_loader
from torch.utils.data import DataLoader

"""
This script tested image_loader unit
should return a image tensor including images of one patient
"""

dlbcl_data = image_loader.ImageLoadPipe(tumor_csv = 'D:/data/lymphoma/closed_seg/DLBCL/selected_VOI.xls',
                           img_dir = 'D:/data/lymphoma/closed_seg/DLBCL/CT_seg')
dlbcl_loader = DataLoader(dlbcl_data,batch_size = 1, shuffle = True)
imgs = next(iter(dlbcl_loader))
print(imgs.shape)
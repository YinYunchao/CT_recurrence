from numpy import dtype
import torch
from Unet_model import Vnet
from image_loader import ImageLoadPipe
from loss_func import DiceLoss
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
'''
https://cloud.tencent.com/developer/article/1776287
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# device = torch.device("cpu")
scan_path = '/data/p288821/dataset/recurrence/training_scan/CT'
livermask_path = '/data/p288821/dataset/recurrence/training_scan/livermask'
tfsummary_savepath = '/data/p288821/tfsummary'

# scan_path = 'X:/dataset/recurrence/training_scan/CT'
# livermask_path = 'X:/dataset/recurrence/training_scan/livermask'
# tfsummary_savepath = 'c:/Users/yyc13/recurrence/tfsummary'

scan_list = os.listdir(scan_path)
scan_datapipe = ImageLoadPipe(scan_path = scan_path,
                                        scan_list = scan_list,
                                        mask_path = livermask_path,
                                        if_transform = True,
                                        augmentation_list=['rotate','contrast','gaussianNoise','GaussianBlur'])
scan_loader = DataLoader(scan_datapipe,batch_size = 1, shuffle = True)
ScanIter = iter(scan_loader)

model = Vnet(elu=True, in_channel=1, classes = 1)
writer = SummaryWriter(tfsummary_savepath)
running_loss = 0.0
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
loss_function = DiceLoss()
# writer.add_graph(model)
model.to(device)

i = 0
for data in ScanIter:
    img, liver_mask = data
    # print(img.shape)
    # print(liver_mask.shape)
    optimizer.zero_grad()
    img = img.to(device,dtype = torch.float)
    liver_mask = liver_mask.to(device, dtype = torch.float)
    prediction = model(img)
    # print(prediction.shape)
    loss = loss_function(prediction,liver_mask)
    # print(loss)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()

    if i%10==0:
        print(loss)
        writer.add_scalars("training loss",{"training":running_loss})
    i+=1
writer.flush()

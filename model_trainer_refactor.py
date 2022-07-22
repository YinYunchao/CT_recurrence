import torch
from Unet_model import Vnet
from image_loader import ImageLoadPipe
from loss_func import DiceLoss
import os
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from train_manager import TrainManager



class TrainBuilder():
    '''
    this class build the model trainer, 
    taking two kwargs: one for read/write path, one for hyper parameters
    '''
    def __init__(self, paths, kwargs):
        self.dataloader = DataLoader(ImageLoadPipe(scan_path=paths['train_scan'],
                                                   scan_list = os.listdir(paths['train_scan']),
                                                   mask_path=paths['train_label'],
                                                   if_transform=True,
                                                   augmentation_list=['rotate','contrast',
                                                                      'gaussianNoise','GaussianBlur']),
                                     batch_size = 1, shuffle = True)
        self.val_dataloader = DataLoader(ImageLoadPipe(scan_path = paths['val_scan'],
                                                    scan_list=os.listdir(paths['val_scan']),
                                                    mask_path=paths['val_label'],
                                                    if_transform=False,
                                                    augmentation_list=None),
                                    batch_size = 1, shuffle=True)
        self.kwargs = kwargs
        self.model = Vnet(elu=True, in_channel=1, classes=1)
        self.loss_func = DiceLoss()
        self.writer = SummaryWriter(paths['tfsummary'])
        self.device = kwargs['device']
        self.model.to(self.device)
        self.TrainBuilder = TrainBuilder
        self.epoch = kwargs['epoch']
        self.learning_rate = kwargs['lr']
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum=0.9)
        self.TrainManager = TrainManager()
        
    
    def run(self):
        '''
        the epoch loop, consist of training loop and validation loop
        '''
        self.TrainManager.begin_run()
        for e_ind in range(self.epoch):
            running_loss = 0.0
            i=0
            for data in self.dataloader:
                img, liver_mask = data
                self.optimizer.zero_grad()
                img = img.to(self.device, dtype = torch.float)
                liver_mask = liver_mask.to(self.device, dtype = torch.float)
                prediction = self.model(img)
                loss = self.loss_func(prediction,liver_mask)
                print(loss)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                i+=1
                if i%10==0:
                    # print(running_loss/i)
                    self.writer.add_scalars("training loss",
                                            {'training':running_loss},
                                            i+e_ind*len(self.dataloader))
            self.model.train(False)
            val_running_loss = 0.0
            for val_data in self.val_dataloader:
                val_img,val_mask = val_data
                val_img = val_img.to(self.device)
                val_mask = val_mask.to(self.device)
                val_prediction = self.model(val_img)
                val_loss = self.loss_func(val_prediction,val_mask)
                val_running_loss += val_loss.detach()
            print(val_running_loss)
            self.writer.add_scalar("training loss",
                                            {'validation':val_running_loss},
                                            i+e_ind*len(self.dataloader))
                
        self.writer.flush()


path_cluster = {'train_scan':'/data/p288821/dataset/recurrence/training_scan/CT',
                'train_label':'/data/p288821/dataset/recurrence/training_scan/livermask',
                'tfsummary':'/data/p288821/tfsummary',
                'val_scan':'/data/p288821/dataset/recurrence/validate_scan/CT',
                'val_label':'/data/p288821/dataset/recurrence/validate_scan/livermask'}
path_local = {'train_scan':'X:/dataset/recurrence/training_scan/CT',
              'train_label':'X:/dataset/recurrence/training_scan/livermask',
              'tfsummary':'c:/Users/yyc13/recurrence/tfsummary',
              'val_scan':'X:/dataset/recurrence/validate_scan/CT',
              'val_label':'X:/dataset/recurrence/validate_scan/livermask'}
params = {'device':'cpu',
          'lr':0.001,
          'epoch':2}


TrainBuilder(paths = path_local,kwargs=params)



        


        

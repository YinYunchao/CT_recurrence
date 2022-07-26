
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
        self.tf_path = paths['tfsummary']
        self.result_savepath = paths['result']
        self.model_SavePath = paths['model']
        self.dataloader = DataLoader(ImageLoadPipe(scan_path=paths['train_scan'],
                                                   scan_list = os.listdir(paths['train_scan']),
                                                   mask_path=paths['train_label'],
                                                   slice_num = 64,
                                                   if_transform=True,
                                                   augmentation_list=['rotate',
                                                                      'GaussianBlur',
                                                                      'EqualHist',
                                                                      'gaussianNoise']),
                                     batch_size = 1, shuffle = True)
        self.val_dataloader = DataLoader(ImageLoadPipe(scan_path = paths['val_scan'],
                                                    scan_list=os.listdir(paths['val_scan']),
                                                    mask_path=paths['val_label'],
                                                    slice_num=64,
                                                    if_transform=False,
                                                    augmentation_list=None),
                                    batch_size = 1, shuffle=True)

        self.kwargs = kwargs
        self.epoch = kwargs['epoch']
        self.device = kwargs['device']
        self.learning_rate = kwargs['lr']

        self.model = Vnet(elu=True, in_channel=1, classes=1)
        self.loss_func = DiceLoss()
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.learning_rate, momentum=0.9)

        self.TrainManager = TrainManager()
        
    
    def run(self):
        '''
        the epoch loop, consist of training loop and validation loop
        '''
        self.TrainManager.begin_run(self.kwargs,self.model,self.dataloader,self.tf_path,self.device)
        for e_ind in range(self.epoch):
            self.TrainManager.begin_epoch()
            i=0
            for data in self.dataloader:
                img, liver_mask = data
                self.TrainManager.sum_img(img,i)
                self.optimizer.zero_grad()
                img = img.to(self.device, dtype = torch.float)
                liver_mask = liver_mask.to(self.device, dtype = torch.float)
                prediction = self.model(img)
                loss = self.loss_func(prediction,liver_mask)
                loss.backward()
                self.optimizer.step()
                self.TrainManager.track_loss(loss, i)
                i+=1

            self.model.train(False)
            self.TrainManager.begin_validation()
            for val_data in self.val_dataloader:
                val_img,val_mask = val_data
                val_img = val_img.to(self.device)
                val_mask = val_mask.to(self.device)
                val_prediction = self.model(val_img)
                val_loss = self.loss_func(val_prediction,val_mask)
                self.TrainManager.track_val_loss(val_loss)
            self.TrainManager.end_validation(len(self.val_dataloader))
            self.TrainManager.save_best_model(self.model_SavePath)
            self.TrainManager.end_epoch()
            self.model.train(True)
            self.TrainManager.save_result_csv(self.result_savepath,'train_result')
        self.TrainManager.end_run()



path_cluster = {'train_scan':'/data/p288821/dataset/recurrence/training_scan/CT',
                'train_label':'/data/p288821/dataset/recurrence/training_scan/livermask',
                'val_scan':'/data/p288821/dataset/recurrence/validate_scan/CT',
                'val_label':'/data/p288821/dataset/recurrence/validate_scan/livermask',
                'tfsummary':'/data/p288821/tfsummary/recurrence',
                'result': '/data/p288821/result/recurrence',
                'model':'/data/p288821/result/recurrence'}

path_local = {'train_scan':'X:/dataset/recurrence/training_scan/CT',
              'train_label':'X:/dataset/recurrence/training_scan/livermask',
              'val_scan':'X:/dataset/recurrence/validate_scan/CT',
              'val_label':'X:/dataset/recurrence/validate_scan/livermask',
              'tfsummary':'c:/Users/yyc13/recurrence/tfsummary',
              'result':'c:/Users/yyc13/recurrence/result',
              'model':'c:/Users/yyc13/recurrence/result'}
params = {'device':'cpu',#torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          'lr':0.0001,
          'epoch':100}


TrainBuilder(paths = path_cluster, kwargs=params).run()


        


        

import time
import os
from collections import OrderedDict
from imageio import save
import pandas as pd
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

class TrainManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None

        self.val_running_loss = 0
        self.val_start_time = None
        self.val_data = []
        self.val_loss = []

        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        
        self.network = None
        self.loader = None
        self.tf_writer = None
    
    def begin_run(self,run,network,loader,tfsummary_path):
        '''
        Execute at the begining of training, including:
        --count time,
        --generate tf summary
        --summary the first image and label from loader, for chekcing img and mask
        --summary the graph of model
        params:
        run: the hyper parameter used in this training run
        network: the model
        loader: the loader
        tfsummary_path: the folder(absolute path) to save tf record 9
        '''
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count+=1
        self.network = network
        self.loader = loader
        self.tf_writer = SummaryWriter(tfsummary_path)

        img,label = next(iter(self.loader))
        grid = torchvision.utils.make_grid([img[0,0,:,:,int(img.shape[-1]/2)],
                                            label[0,0,:,:,int(img.shape[-1]/2)]])
        self.tf_writer.add_image('img_check',grid)
        self.tf_writer.add_graph(self.network,img)

    def end_run(self):
        self.tf_writer.flush()
        self.tf_writer.close()
        self.epoch_count = 0
        self.epoch_num_correct = 0
    
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count+=1
        self.epoch_loss = 0
    
    def end_epoch(self):
        epoch_duration= time.time()-self.epoch_start_time
        run_duration = time.time()-self.run_start_time

        loss = self.epoch_loss/len(self.loader)
        # accuracy = self.epoch_num_correct / len(self.loader)
        self.tf_writer.add_scalar('epoch_loss', loss, self.epoch_count)
        # self.tf_writer.add_scalar('epoch_acc', accuracy, self.epoch_count)
        if self.epoch_count%5==0:
            for name, param in self.network.named_parameters():
                self.tf_writer.add_histogram(name, param, self.epoch_count)
                self.tf_writer.add_histogram(f'{name}.grad',param.grad,self.epoch_count)
        results = OrderedDict()
        # results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        # results["accuracy"] = accuracy
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration
        for key in self.run_params: results[key] = self.run_params[key]
        self.run_data.append(results)
        # df = pd.DataFrame.from_dict(self.run_data,orient='columns')

    def track_loss(self,loss,step):
        self.epoch_loss+=loss.item()
        self.tf_writer.add_scalar('step_loss', loss.item(), step+(self.epoch_count-1)*len(self.loader))

    def track_num_correct(self, prediction, label):
        self.epoch_num_correct+=self.get_num_correct(prediction, label)

    def _get_num_correct(self, prediction, labels):
        '''
        only used for classification tasks
        '''
        return prediction.argmax(dim=1).eq(labels).sum().item()

    def begin_validation(self):
        self.val_running_loss = 0
        self.val_start_time = time.time()

    def track_val_loss(self, val_loss):
        self.val_running_loss+=val_loss.item()
        
    def end_validation(self,val_dataNum):
        val_results = OrderedDict()
        val_results["epoch"] = self.epoch_count
        val_results["loss"] = self.val_running_loss/val_dataNum
        # val_results["accuracy"] = accuracy
        val_results["epoch_duration"] = time.time()-self.val_start_time
        #print('time for validation: ', val_results['epoch_duration'], ';',val_results['loss'])
        self.val_running_loss = self.val_running_loss/val_dataNum
        self.val_loss.append(val_results["loss"])
        self.val_data.append(val_results)
        self.tf_writer.add_scalar('epoch{}_validation_loss:'.format(self.epoch_count),
                                    self.val_running_loss,self.epoch_count)
        
    def save_result_csv(self, save_path, fileName):
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(os.path.join(save_path,f'{fileName}.csv'))
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(os.path.join(save_path,'val_{}.csv'.format(fileName)))

    def save_best_model(self, save_path):
        '''
        The model will be saved if the validation loss is the smallest among all epochs
        '''
        if self.val_running_loss < min(self.val_loss):
            torch.save(self.model.state_dict(),save_path)

    def load_saved_model(self,load_path):
        model = torch.load(load_path)
        model.eval()




    


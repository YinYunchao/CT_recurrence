import time
from collections import OrderedDict
import pandas as pd
import torchvision
from torch.utils.tensorboard import SummaryWriter

class TrainManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_start_time = None

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
        tfsummary_path: the folder(absolute path) to save tf record
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
        accuracy = self.epoch_num_correct / len(self.loader)
        self.tf_writer.add_scalar('epoch_loss', loss, self.epoch_count)
        self.tf_writer.add_scalar('epoch_acc', accuracy, self.epoch_count)
        if self.epoch_count%5==0:
            for name, param in self.network.named_parameters():
                self.tf_writer.add_histogram(name, param, self.epoch_count)
                self.tf_writer.add_histogram(f'{name}.grad',param.grad,self.epoch_count)
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = loss
        results["accuracy"] = accuracy
        results["epoch_duration"] = epoch_duration
        results["run_duration"] = run_duration
        for k,v in self.run_params._asdict().items():results[k] = v
        self.run_data.append(results)
        # df = pd.DataFrame.from_dict(self.run_data,orient='columns')

    def track_loss(self,loss,batch):
        self.epoch_loss+=loss.item()*batch[0].shape[0]
    def track_num_correct(self, prediction, label):
        self.epoch_num_correct+=self.get_num_correct(prediction, label)

    def _get_num_correct(self, prediction, labels):
        return prediction.argmax(dim=1).eq(labels).sum().item()
    def save(self, fileName):

        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')


    


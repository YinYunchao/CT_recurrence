from distutils.log import debug
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
'''
This is a standard Vnet model, the code is based on the Github pages below
Modification was made based on a pre-trained Vnet model by UMCG Data 
Reference:
#https://github.com/jaxony/unet-pytorch/blob/master/model.py
#https://theaisummer.com/unet-architectures/#v-net-2016
#https://github.com/Dawn90/V-Net.pytorch/blob/master/vnet.py
the code is refactored as the link above, as it's neat and efficient 
'''
def passthrough(img,**kwargs):
    return img

def _make_nConv(nchan, depth, elu):
    '''
    this func generate the convolutional layers by loop

    param:
    nchan: input channel number
    depth: number of convolutional layers generated, loop times
    elu: param for activation func choice
    '''
    layers = []
    for _ in range(depth):
        layers.append(conv_comblayer(nchan=nchan, elu=elu))
    return nn.Sequential(*layers)

def activation_func(elu, nchan):
    '''
    offer two choices of activation function
    '''
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

class conv_comblayer(nn.Module):
    '''
    the combination components of convolutional layers
    including:conv layer, activation func, batch normalization
    '''
    def __init__(self, nchan, elu):
        super(conv_comblayer, self).__init__()
        self.relu1 = activation_func(elu=elu, nchan=nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm3d(nchan)
    def forward(self, img):
        out = self.relu1(self.bn1(self.conv1(img)))
        return out



class InputLayer(nn.Module):
    '''
    the first layer of Vnet
    '''
    def __init__(self, in_channel, elu, out_channels = 16):
        super(InputLayer,self).__init__()
        self.out_channels = out_channels
        self.in_channel = in_channel  
        self.conv1 = nn.Conv3d(1,out_channels = out_channels,kernel_size=5,padding = 'same')
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = activation_func(elu,out_channels)
    def forward(self,img):
        out = self.conv1(img)
        out = self.bn1(out)
        repeat_rate = int(self.out_channels/self.in_channel) #for rgb image sake, not necessary divide
        img_re16 = img.repeat(1,repeat_rate,1,1,1)
        return self.relu1(torch.add(out,img_re16))

class encode_layer(nn.Module):
    '''
    this func combines the components above to generate one layer of encoder
    including down-sampling by stride conv, batch normalization, dropout and layers of convolution on request
    '''
    def __init__(self, in_channel, nConvs, elu, dropout = False):
        super(encode_layer, self).__init__()
        out_channel = in_channel*2
        self.down_conv = nn.Conv3d(in_channels=in_channel, out_channels=out_channel,
                            kernel_size=2, stride=2) #this is pooling layer, not extracting features
        self.bn1 = nn.BatchNorm3d(out_channel)

        self.dropout_layer1 = passthrough
        self.relu1 = activation_func(elu=elu, nchan=out_channel)
        self.relu2 = activation_func(elu=elu, nchan=out_channel)
        if dropout:
            self.dropout_layer1 = nn.Dropout3d()
        self.conv_gen = _make_nConv(out_channel, depth=nConvs,elu=elu)
    def forward(self, img):
        downsampled_img = self.relu1(self.bn1(self.down_conv(img)))
        out = self.dropout_layer1(downsampled_img)
        out = self.conv_gen(out)#generate the conv layers
        out = self.relu2(torch.add(out,downsampled_img)) #add the residual images
        return out

class decode_layer(nn.Module):
    '''
    this fund combines the components above to generate a decoder
    '''
    def __init__(self, in_channel, out_channel, nConvs, elu, dropout = False):
        super(decode_layer,self).__init__()
        self.up_conv = nn.ConvTranspose3d(in_channels=in_channel,out_channels=out_channel//2,
                            kernel_size=2, stride=2)
        self.bn1 = torch.nn.BatchNorm3d(out_channel//2)
        self.dropout1 = passthrough
        self.dropout2 = nn.Dropout3d()
        self.relu1 = activation_func(elu, out_channel//2)
        self.relu2 = activation_func(elu, out_channel)
        if dropout:
            self.dropout1 = nn.Dropout3d()
        self.conv_gen = _make_nConv(out_channel, nConvs, elu)
    def forward(self, img, skiped_img):
        img_out = self.dropout1(img)
        skimg_out = self.dropout2(skiped_img)
        img_out = self.relu1(self.bn1(self.up_conv(img_out)))
        img_cat = torch.cat((img_out,skimg_out),1)
        img_out = self.conv_gen(img_cat)
        img_out = self.relu2(torch.add(img_out,img_cat))
        return img_out

class output_layer(nn.Module):
    '''
    the layer before output, the out_channel equals to the classes required by task
    '''
    def __init__(self, in_channel, classes, elu):
        super(output_layer, self).__init__()
        self.classes = classes
        self.conv1 = nn.Conv3d(in_channels = in_channel,out_channels=classes,kernel_size=1)
        self.bn1 = torch.nn.BatchNorm3d(classes)
        self.conv2 = nn.Conv3d(in_channels=classes,out_channels=classes, kernel_size=1)
        self.relu1 = activation_func(elu,classes)
    def forward(self,img):
        out_img = self.relu1(self.bn1(self.conv1(img)))
        out_img = self.conv2(out_img)
        return out_img

class Vnet(nn.Module):
    '''
    build the Vnet based on the component above,
    the name of each encode/decode layer was named according to the output channel number
    '''
    def __init__(self,elu = True, in_channel=1, classes = 2):
        super(Vnet, self).__init__()
        self.classes = classes
        self.in_channel = in_channel
        self.input_layer = InputLayer(in_channel=in_channel,elu=elu,out_channels=8)
        self.encode_ch32 = encode_layer(in_channel=8, nConvs=1, elu=elu, dropout = False)
        self.encode_ch64 = encode_layer(in_channel=16, nConvs=2, elu=elu, dropout=False)
        self.encode_ch128 = encode_layer(in_channel=32, nConvs=3, elu=elu, dropout=True)
        self.encode_ch256 = encode_layer(in_channel=64, nConvs=2, elu=elu, dropout=True)
        self.decode_ch256 = decode_layer(in_channel = 128, out_channel = 128, nConvs = 2,
                                         elu = elu, dropout = True)
        self.decode_ch128 = decode_layer(in_channel=128, out_channel=64, nConvs=2, elu=elu, dropout=True)
        self.decode_ch64 = decode_layer(in_channel=64, out_channel=32, nConvs=1, elu=elu,dropout=False)
        self.decode_ch32 = decode_layer(in_channel=32, out_channel=16, nConvs=1, elu=elu, dropout=False)
        self.output_layer = output_layer(in_channel = 16, classes = classes, elu = elu)

    def forward(self, img):
        enimg_ch16 = self.input_layer(img) # encoded image, with 16 channels; 
        #the output is stored in different variable as residual img used in decode layers 
        enimg_ch32 = self.encode_ch32(enimg_ch16)
        enimg_ch64 = self.encode_ch64(enimg_ch32)
        enimg_ch128 = self.encode_ch128(enimg_ch64)
        enimg_ch256 = self.encode_ch256(enimg_ch128)
        de_img = self.decode_ch256(enimg_ch256,enimg_ch128) #decoded image
        de_img = self.decode_ch128(de_img, enimg_ch64)
        de_img = self.decode_ch64(de_img, enimg_ch32)
        de_img = self.decode_ch32(de_img, enimg_ch16)
        de_img = self.output_layer(de_img)
        return de_img

    def test(self, device = 'cpu'):
        '''
        to test whether bugs in the model builded above, input is randomly generated
        '''
        input_tensor = torch.rand(1, self.in_channel,512,512,32)
        label = torch.rand(1,self.classes, 512,512,32)
        out = self.forward(input_tensor)
        assert label.shape==out.shape
        summary(self.to(torch.device(device=device)),(self.in_channel,512,512,32),device=device)
        print("Vnet test is complete")
        
        
        

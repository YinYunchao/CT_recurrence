from torch import rand
from Unet_model import Vnet
import numpy as np


obj = Vnet(elu=True,in_channel=1,classes=1)
obj.test(device='cpu')
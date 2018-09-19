import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import os 
import sys
from glob import glob
import cv2


class model_wrapper(nn.Module):
    def __init__(self, model, channels_in=1,channels_out=2):
        super(model_wrapper, self).__init__()
        self.model = model
        self.one2three = nn.Sequential(
                nn.Conv2d(channels_in,3,kernel_size=1,padding=0))
        self.linear = nn.Sequential(
                nn.Linear(1000,channels_out))
    def forward(self, x):
        # out = self.one2three(x)
        out = self.model(x)
        out = self.linear(out)
        return out



pre_trained = sys.argv[1]
pre_weight = torch.load(pre_trained)

# load pretrained weight
NN = models.resnet18()
model = model_wrapper(NN,3,53)
model.load_state_dict(pre_weight)

input_path='/home/ej/Desktop/eyenet_ACCV/Age related macular degeneration'
imgs_name = glob(input_path+'/**/*.jpg',recursive = True)
print(input_path,len(imgs_name))
images = []
for i in imgs_name:
    try:
        ii = cv2.imread(i,cv2.IMREAD_COLOR)
        iii = cv2.resize(ii,(224,224))
        iii = np.reshape(iii,(3,224,224))
        images.append(iii)
    except:
        print(i)
 
X_train = np.array(images)
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
ipt = torch.from_numpy(X_train)
ipt = ipt.float()

out = model(ipt[:10])
pred = torch.max(out,1)[1].data.numpy().squeeze()

print(pred)

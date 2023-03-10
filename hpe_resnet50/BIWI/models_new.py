import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchsummary import summary

import pickle

class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000,num_bits=10,code='u',init="rand",reqgrad=0.0):
        print(block.expansion)
        print(layers)
        code_bits={'j2':[150,100],'u':[150,100],'sim':[150,100],'auto':[150,100],'j':[75,50],'b1jdj':[39,26],'b2jdj':[21,15],'hex16':[16,16]}
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        if num_bits==0:
            self.fc_angles_yaw = nn.Linear(512 * block.expansion, code_bits[code][0])
            self.fc_angles_pitch = nn.Linear(512 * block.expansion, code_bits[code][0])
            self.fc_angles_roll = nn.Linear(512 * block.expansion, code_bits[code][1])
        else:
            self.fc_angles_yaw = nn.Sequential(nn.Linear(512 * block.expansion, num_bits),nn.Linear(num_bits, code_bits[code][0]))
            self.fc_angles_pitch = nn.Sequential(nn.Linear(512 * block.expansion, num_bits),nn.Linear(num_bits, code_bits[code][0]))
            self.fc_angles_roll = nn.Sequential(nn.Linear(512 * block.expansion, num_bits),nn.Linear(num_bits, code_bits[code][1]))
        
        di=pickle.load(open("code_pkl/bel"+code+"_150_tensor.pkl","rb"))
        dis=pickle.load(open("code_pkl/bel"+code+"_100_tensor.pkl","rb"))


        di=torch.transpose(di,0,1).cuda()
        dis=torch.transpose(dis,0,1).cuda()

        if init=="p1":
	        self.yawm = torch.nn.Parameter(torch.ones(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
	        self.pitchm = torch.nn.Parameter(torch.ones(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
        	self.rollm = torch.nn.Parameter(torch.ones(size=(code_bits[code][1], 100), dtype=torch.float, requires_grad=True).cuda())
        #p0
        elif init=="p0":
        	self.yawm = torch.nn.Parameter(torch.zeros(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
        	self.pitchm = torch.nn.Parameter(torch.zeros(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
        	self.rollm = torch.nn.Parameter(torch.zeros(size=(code_bits[code][1], 100), dtype=torch.float, requires_grad=True).cuda())
        #01rand
        elif init=="rand":
        	self.yawm = torch.nn.Parameter(torch.rand(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
        	self.pitchm = torch.nn.Parameter(torch.rand(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
        	self.rollm = torch.nn.Parameter(torch.rand(size=(code_bits[code][1], 100), dtype=torch.float, requires_grad=True).cuda())
        #norm
        #self.code = torch.nn.Parameter(torch.empty(size=(config.CODE.CODE_BITS, config.BITS), dtype=torch.float, requires_grad=True).cuda())
        #torch.nn.init.normal_(self.code, mean=0.0, std=0.25)
        else:
	        self.yawm = torch.nn.Parameter(torch.empty(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
	        self.pitchm = torch.nn.Parameter(torch.empty(size=(code_bits[code][0], 150), dtype=torch.float, requires_grad=True).cuda())
	        self.rollm = torch.nn.Parameter(torch.empty(size=(code_bits[code][1], 100), dtype=torch.float, requires_grad=True).cuda())
        	with torch.no_grad():
        	    self.yawm.copy_(di)
        	    self.pitchm.copy_(di)
        	    self.rollm.copy_(dis)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        yawi = self.fc_angles_yaw(x)
        yaw = torch.matmul(yawi,self.yawm)
        pitchi = self.fc_angles_pitch(x)
        pitch = torch.matmul(pitchi,self.pitchm)

        rolli = self.fc_angles_roll(x)
        roll = torch.matmul(rolli,self.rollm)

        return yaw,pitch,roll,1,yawi,pitchi,rolli #torch.cat((y,p,r),dim=1)


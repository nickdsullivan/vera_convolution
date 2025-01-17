'''
ResNet-50 Architecture.
'''

import torch
import torch.nn as nn
from .cbam import CBAM
from loralib.layers import ConvLoRA
from loralib import utils

class BottleNeck(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, in_channels, out_channels, expansion=1, stride=1, use_cbam=True,use_vera=True,use_lora=False):
        '''Param init.
        '''
        super(BottleNeck, self).__init__()
        self.use_vera = use_vera or use_lora
        self.use_cbam = use_cbam
        #only the first conv will be affected by the given stride parameter. The rest have default stride value (which is 1).
        if self.use_vera:
            self.conv1 = ConvLoRA(nn.Conv2d, in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1,r=4, use_vera = use_vera)
            self.conv1.name= "lora1"
            self.conv2 = ConvLoRA(nn.Conv2d, in_channels=out_channels, out_channels=out_channels,padding = 1,kernel_size=3,r=4, use_vera = use_vera)
            self.conv1.name= "lora2"
            self.conv3 = ConvLoRA(nn.Conv2d, in_channels=out_channels, out_channels=out_channels*expansion,kernel_size=1,r=4, use_vera = use_vera)
            self.conv1.name= "conv3"
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, stride=stride)
            self.conv1.name= "conv1"
            self.bn1 = nn.BatchNorm2d(num_features=out_channels)
            self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding = 1, bias=False)
            self.conv1.name= "conv2"
            self.bn2 = nn.BatchNorm2d(num_features=out_channels)
            self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels*expansion, kernel_size=1, bias=False)
            self.conv1.name= "conv3"
            self.bn3 = nn.BatchNorm2d(num_features=out_channels*expansion)
            self.relu = nn.ReLU(inplace=True)
        
        #since the input has to be same size with the output during the identity mapping, whenever the stride or the number of output channels are
        #more than 1 and expansion*out_channels respectively, the input, x, has to be downsampled to the same level as well.
        self.identity_connection = nn.Sequential()
        if stride != 1 or in_channels != expansion*out_channels:
            self.identity_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels*expansion)
            )

        if self.use_cbam:
            self.cbam = CBAM(channel_in=out_channels*expansion)


    def forward(self, x):
        '''Forward Propagation.
        '''

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))

        
        if self.use_cbam:
            out = self.cbam(out)
        #print(out.shape)
        out += self.identity_connection(x) #identity connection/skip connection
        out = self.relu(out)
        

        return out


class ResNet50(nn.Module):
    '''ResNet-50 Architecture.
    '''

    def __init__(self, use_cbam=True, use_vera=True, use_lora=False, image_depth=3, num_classes=6,img_size=224):
        '''Params init and build arch.
        '''
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.expansion = 4
        self.num_blocks = [3, 3, 3, 2]
        self.img_size_multiplier = max(int(img_size//28)-1,1)
        self.conv_block1 = nn.Sequential(nn.Conv2d(kernel_size=self.img_size_multiplier, stride=2, in_channels=image_depth, out_channels=self.in_channels, padding=3, bias=False),
                                            nn.BatchNorm2d(self.in_channels),
                                            nn.ReLU(inplace=True),
                                            nn.MaxPool2d(stride=2, kernel_size=3, padding=1))
        
        self.layer1 = self.make_layer(out_channels=64, num_blocks=self.num_blocks[0], stride=1, use_cbam=use_cbam, use_lora=use_lora, use_vera=use_vera)
        self.layer2 = self.make_layer(out_channels=128, num_blocks=self.num_blocks[1], stride=2, use_cbam=use_cbam, use_lora=use_lora, use_vera=use_vera)
        self.layer3 = self.make_layer(out_channels=256, num_blocks=self.num_blocks[2], stride=2, use_cbam=use_cbam, use_lora=use_lora, use_vera=use_vera)
        self.layer4 = self.make_layer(out_channels=512, num_blocks=self.num_blocks[3], stride=2, use_cbam=use_cbam, use_lora=use_lora, use_vera=use_vera)
        
        #print(self.layer1.numel())
        self.avgpool = nn.AvgPool2d(self.img_size_multiplier)
        self.linear = nn.Linear(512*self.expansion, num_classes)
        #self.linear = nn.Linear(512*16, num_classes)


    def make_layer(self, out_channels, num_blocks, stride, use_cbam, use_vera,use_lora):
        '''To construct the bottleneck layers.
        '''
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BottleNeck(in_channels=self.in_channels, out_channels=out_channels, stride=stride, expansion=self.expansion, use_cbam=use_cbam, use_vera=use_vera, use_lora=use_lora))
            self.in_channels = out_channels * self.expansion
        return nn.Sequential(*layers)


    def forward(self, x):
        '''Forward propagation of ResNet-50.
        '''

        x = self.conv_block1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_conv = self.layer4(x)
        x = self.avgpool(x_conv)
        x = nn.Flatten()(x) #flatten the feature maps.
        x = self.linear(x)

        return x_conv, x


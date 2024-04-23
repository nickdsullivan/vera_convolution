#testmodel

import torch
import torch.nn as nn
from loralib.layers import ConvLoRA
from loralib import utils

import os
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torchsummary import summary
from torchvision import transforms
import torch.distributed as dist
#import torch.multiprocessing as mp

from models.resnet50 import ResNet50
from runtime_args import args
from load_dataset import LoadDataset
from plot import plot_loss_acc
from helpers import calculate_accuracy
import torchvision.datasets as datasets

import torchvision.transforms as transforms
class test_model(nn.Module):
    '''Bottleneck modules
    '''

    def __init__(self, in_channels, out_channels, expansion=1, stride=1, use_cbam=False,use_vera=True,use_lora=False):
        '''Param init.
        '''
        super(test_model, self).__init__()
        use_vera = True
        self.use_vera = use_vera or use_lora
        self.use_cbam = use_cbam
        #only the first conv will be affected by the given stride parameter. The rest have default stride value (which is 1).
        if self.use_vera:
            self.conv1 = ConvLoRA(nn.Conv2d, in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=1,r=16, use_vera = use_vera)
            self.conv1.name= "lora1"
            self.conv2 = ConvLoRA(nn.Conv2d, in_channels=out_channels, out_channels=out_channels,padding = 1,kernel_size=3,r=16, use_vera = use_vera)
            self.conv1.name= "lora2"
            self.conv3 = ConvLoRA(nn.Conv2d, in_channels=out_channels, out_channels=out_channels*expansion,kernel_size=1,r=16, use_vera = use_vera)
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
        out = nn.Flatten()(out)

        return out





















def test(args):
    transform = transforms.Compose([
            transforms.Resize((244,244)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            ])

    model = test_model(in_channels=3, out_channels=1, use_cbam=args.use_cbam,use_vera = args.use_vera, use_lora = args.use_lora)
    #torch.cuda.set_device(gpu)
    #model.cuda(gpu)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_rate)
    #criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    criterion = torch.nn.CrossEntropyLoss()



    summary(model, (3, 224, 224))


    #train_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=True,
    #transform=transforms.ToTensor())
    #test_dataset = LoadDataset(dataset_folder_path=args.data_folder, image_size=args.img_size, image_depth=args.img_depth, train=False,
    #transform=transforms.ToTensor())


    train_dataset = datasets.MNIST(root=args.data_folder, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(root=args.data_folder, train=False, download=False, transform=transform)

    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_generator = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler)
    test_generator = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    for i, sample in tqdm(enumerate(train_generator)):
        batch_x, batch_y = sample[0], sample[1]
        print(model(batch_x))
        #print(new_output)
if __name__ == '__main__':
    test(args)

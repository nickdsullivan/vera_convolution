'''Training script.
'''


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

# device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'gpu' else 'cpu')

if not os.path.exists(args.graphs_folder) : os.mkdir(args.graphs_folder)
model_save_folder = 'resnet_cbam/' if args.use_cbam else 'resnet/'

if not os.path.exists(model_save_folder) : os.mkdir(model_save_folder)



def train(gpu, args):
    '''Init models and dataloaders and train/validate model.
    '''
    print("init")

    transform = transforms.Compose([
        transforms.Resize((args.img_size,args.img_size)),
        transforms.Grayscale(num_output_channels=args.img_depth),
        transforms.ToTensor(),
        ])

    model = ResNet50(image_depth=args.img_depth, num_classes=args.num_classes, use_cbam=args.use_cbam,use_vera = args.use_vera, use_lora = args.use_lora,img_size=args.img_size)
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


    training_loss_list = []
    training_acc_list = []
    testing_loss_list = []
    testing_acc_list = []

    best_accuracy = 0
    print(len(train_generator))
    print("***************************")
    print("Number of trainable parameters")
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("***************************")
    #print(p.requires_grad)
    for epoch_idx in range(args.epoch):
        #Model Training & Validation.
        model.train()

        epoch_loss = []
        epoch_accuracy = []
        i = 0
        for i, sample in tqdm(enumerate(train_generator)):
            #batch_x, batch_y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)

            
            batch_x, batch_y = sample[0], sample[1]
            

            optimizer.zero_grad()

            _,net_output = model(batch_x)
            total_loss = criterion(input=net_output, target=batch_y)

            total_loss.backward()
            optimizer.step()
            batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
            epoch_loss.append(total_loss.item())
            epoch_accuracy.append(batch_accuracy)

        curr_accuracy = sum(epoch_accuracy)/(i+1)
        curr_loss = sum(epoch_loss)/(i+1)

        training_loss_list.append(curr_loss)
        training_acc_list.append(curr_accuracy)

        print(f"Epoch {epoch_idx}")
        print(f"Training Loss : {curr_loss}, Training accuracy : {curr_accuracy}")

        model.eval()
        epoch_loss = []
        epoch_accuracy = []
        i = 0

        with torch.set_grad_enabled(False):
            for i, sample in tqdm(enumerate(test_generator)):

                #batch_x, batch_y = sample['image'].cuda(non_blocking=True), sample['label'].cuda(non_blocking=True)
                batch_x, batch_y = sample[0], sample[1]

                _,net_output = model(batch_x)

                total_loss = criterion(input=net_output, target=batch_y)

                batch_accuracy = calculate_accuracy(predicted=net_output, target=batch_y)
                epoch_loss.append(total_loss.item())
                epoch_accuracy.append(batch_accuracy)

            curr_accuracy = sum(epoch_accuracy)/(i+1)
            curr_loss = sum(epoch_loss)/(i+1)

            testing_loss_list.append(curr_loss)
            testing_acc_list.append(curr_accuracy)

        print(f"Testing Loss : {curr_loss}, Testing accuracy : {curr_accuracy}")

        #plot accuracy and loss graph
        plot_loss_acc(path=args.graphs_folder, num_epoch=epoch_idx, train_accuracies=training_acc_list, train_losses=training_loss_list,
                            test_accuracies=testing_acc_list, test_losses=testing_loss_list)

        if epoch_idx % 5 == 0:

            lr_decay.step() #decrease the learning rate at every n epoch.
            curr_lr = 0
            for params in optimizer.param_groups:
                curr_lr = params['lr']
            print(f"The current learning rate for training is : {curr_lr}")


        if best_accuracy < curr_accuracy:
            torch.save(model.state_dict(), f"{model_save_folder}model.pth")
            best_accuracy = curr_accuracy
            print('Model is saved!')


        print('\n--------------------------------------------------------------------------------\n')

os.environ['MASTER_ADDR'] = '10.106.15.226'
os.environ['MASTER_PORT'] = '8888'

if __name__ == '__main__':
    train(0, args)
    #train(nprocs,)


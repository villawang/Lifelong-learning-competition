import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
import torch.utils.data
from torchvision import transforms
from tqdm import tqdm_notebook, tnrange
import pickle
from MobileNetV2 import MobileNetV2
import glob
import pdb

# %config InlineBackend.figure_format = 'retina'




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%%
# load mobilenet without the last fc layer
net = MobileNetV2(n_class=69)
if torch.cuda.is_available():
#     net = net.cuda()
    # add map_location='cpu' if no gpu
    loaded_dict = torch.load('./model/mobilenet_v2.pth.tar')
else:
    loaded_dict = torch.load('mobilenet_v2.pth.tar', map_location='cpu')
state_dict = {k: v for k, v in loaded_dict.items() if k in net.state_dict()}
state_dict["classifier.1.weight"] = net.state_dict()["classifier.1.weight"].clone()
state_dict["classifier.1.bias"] = net.state_dict()["classifier.1.bias"].clone()
net.load_state_dict(state_dict)

# freeze parameters
# for name, child in net.named_children():
#     if name == 'features':
#         for para in child.parameters():
#             para.requires_grad = False

total_params = sum(p.numel() for p in net.parameters())
print('total_params:', total_params)
total_trainable_params = sum(p.numel()
                             for p in net.parameters() if p.requires_grad)
print('total_trainable_params:', total_trainable_params)

# determine optimizer
# this loss is used for calculating the performance for both old and new task
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([p for p in net.parameters()][-2:-1], lr=0.005,
                      momentum=0.9, weight_decay=5e-4)

# optimizer for theta_s
optimizer_s = optim.SGD([p for p in net.parameters()][0:-2], lr=0.0005,
                      momentum=0.9, weight_decay=5e-4)

# check if cuda exist then use cuda otherwise use cpu
net=nn.DataParallel(net, device_ids=[0,1])
net.to(device)

# %%
dataset_path = '/home/zhengwei/notebooks/dataset/lifelong_final/'
# load our label for each class
with open('./output/idx_to_label.pickle', 'rb') as pfile:
    idx_to_label = pickle.load(pfile)


def GenerateLabel(path, datatype):
    path_prefix = os.path.join(path, datatype)
    tasks = ['task{}'.format(i) for i in range(1, 13)]
    img_list = []
    label_list = []
    for task in tasks:
        path_prefix_task = os.path.join(path_prefix, task)
        for root, dirs, files in os.walk(path_prefix_task):
            for dirs_ in dirs:
                path_prefix_img = os.path.join(root, dirs_)
                imgs = glob.glob(os.path.join(path_prefix_img, '*'))
                img_list += imgs
                label_list += [dirs_] * len(imgs)
    return img_list, label_list


img_list_train, label_list_train = GenerateLabel(dataset_path, 'train')
img_list_val, label_list_val = GenerateLabel(dataset_path, 'validation')




class BatchData(Dataset):
    #     def format_images(self, path, datatype, batch_index):
    #         path_prefix = '{}/{}/batch{}/'.format(path, datatype, batch_index)
    #         path_prefix = os.path.join(path,datatype,'batch'+str(batch_index))+'/'
    #         table = pd.read_csv(path_prefix + 'label.csv', index_col=0)
    #         data_list = [path_prefix + filename for filename in table['file name'].tolist()]
    #         label_list = table['label'].tolist()
    #         return data_list, label_list

    def format_images(self, path, datatype, batch_index, idx_to_label):
        path_prefix = os.path.join(path, datatype)
        img_list = []
        label_list = []
        path_prefix_task = os.path.join(
            path_prefix, 'task{}'.format(batch_index))
        for root, dirs, files in os.walk(path_prefix_task):
            for dirs_ in dirs:
                path_prefix_img = os.path.join(root, dirs_)
                imgs = glob.glob(os.path.join(path_prefix_img, '*'))
                img_list += imgs
                label_list += [dirs_] * len(imgs)
        mapped_label = [idx_to_label[label] for label in label_list]
        return img_list, mapped_label

    def __init__(self, path, datatype, batch_index, transforms, idx_to_label):
        self.transforms = transforms
        self.data_list, self.label_list = self.format_images(
            path, datatype, batch_index, idx_to_label)

        # print a summary
        print('Load {} batch {} have {} images '.format(
            datatype, batch_index, len(self.data_list)))

    def __getitem__(self, idx):
        img = self.data_list[idx]
        img = Image.open(img)
        label = int(self.label_list[idx])
        img = self.transforms(img)
        return img, label, self.data_list[idx].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.data_list)


# dataset_path = 'lifelong/dataset_final/'
# output_path = './output'
# print(dataset_path, output_path)

# Image preprocessing
trans = transforms.Compose([
    #                     transforms.Resize((300,300)),
    transforms.RandomResizedCrop(255),
    #     transforms.RandomSizedCrop(255),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# keep shuffling be constant every time
seed = 1
torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


train_batch_list = [
    BatchData(dataset_path, 'train', i, trans, idx_to_label) for i in range(1, 13)]
train_loader_list = [torch.utils.data.DataLoader(batch, batch_size=128, shuffle=True, num_workers=1)
                     for batch in train_batch_list]

valid_batch_list = [BatchData(
    dataset_path, 'validation', i, trans, idx_to_label) for i in range(1, 13)]
valid_loader_list = [torch.utils.data.DataLoader(batch, batch_size=128, shuffle=True, num_workers=1)
                     for batch in valid_batch_list]
#%%
# life long without forgetting strategy


def feed(dataloader, is_training, init_task=False,
         num_epochs=50, validloader=None, is_valid=False, is_save_csv=False, csv_name=None,validloader_previous=None):
    losses = list()
    acces = list()
    filename = list()
    label_gt = list()
    label_predict = list()

    since = time.time()
    start = time.time()
    if is_training:
        net.train()
    else:
        num_epochs = 1
        net.eval()
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#--------------------------this is initial task where all parameters needs to be trained--------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
    if init_task == True:
        # tracking acc and loss during training (used for plot)
        loss_train_debug = []
        loss_valid_debug = []
        acc_train_debug = []
        acc_valid_debug = []
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            train_acc = 0.0
            i = 0
            running_losses = []
            acces = []
            for data in dataloader:
                i += 1
                # get the inputs
                inputs, labels, file = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                optimizer_s.zero_grad()
                outputs = net(inputs)
                if is_training:
                    _, pred = outputs.max(1)
                    loss_init = criterion(outputs, labels)
                    loss_init.backward()
                    optimizer.step()
                    optimizer_s.step()

                _, pred = outputs.max(1)
                num_correct = (pred == labels).sum().item()
                acc = num_correct / inputs.shape[0]
                acc_train_debug.append(acc)
                
                # For csv output
                label_predict.extend(pred.tolist())
                label_gt.extend(labels.tolist())
                filename.extend(list(file))
                train_acc = acc
                running_loss = criterion(outputs, labels)

                acces.append(train_acc)
                running_losses.append(running_loss.view(1,1))
#                 loss_train_debug.append(running_loss.data)
#                 outputs.detach()
            running_losses = torch.cat(running_losses)
            time_elapsed = time.time() - since
            since = time.time()

            if is_training:
                valid_acc, valid_loss = valid(validloader)
#                 acc_valid_debug.append(valid_acc)
#                 loss_valid_debug.append(valid_loss)
                print(
                    'epoch{}/{} time:{:.0f}m {:.0f}s train_acc:{:.3f} train_loss:{:.4f} valid_acc:{:.3f} valid loss:{:.3f}'.format(
                        epoch + 1,
                        num_epochs, time_elapsed // 60,
                        time_elapsed % 60,
                        sum(acces) / len(dataloader),
                        running_losses.mean(),
                        valid_acc,
                        valid_loss))
#         pdb.set_trace()
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#--------------------------this is continuous tasks with another loss function--------------------
#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
    else:
        if is_training:
            # cache the theta_o for the old task and copy feature extraction parameters to theta_n
            theta_o = deepcopy(net.state_dict())
            acces, running_losses = train_2(dataloader, validloader, num_epochs,
                                            theta_o, validloader_previous)

    return sum(acces) / len(dataloader), sum(running_losses) / len(dataloader)


def valid(dataloader):
    running_losses = []
    acces = []
    i = 0
    for data in dataloader:
        i += 1
        # get the inputs
        inputs, labels, file = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, pred = outputs.max(1)

        num_correct = (pred == labels).sum().item()

        acc = num_correct / inputs.shape[0]

        running_loss = criterion(outputs, labels)

        acces.append(acc)
        running_losses.append(running_loss.data.view(1,1))
    running_losses = torch.cat(running_losses)
#         outputs.detach()
    return sum(acces) / len(dataloader), running_losses.mean()


# this function applies learning without forgetting strategy to continuous tasks
def train_2(trainloader, validloader, num_epochs, theta_o, validloader_previous=None):
    since = time.time()
    start = time.time()
    step = 0
    # initialize the last layer theta_n
#     theta_n = deepcopy(theta_o)
#     theta_n = deepcopy(net.state_dict())
#     theta_n_bottleneck = torch.autograd.Variable(
#         torch.Tensor(theta_o['module.classifier.1.weight'].size()), requires_grad=True)
#     theta_n_bottleneck_bias = torch.autograd.Variable(
#         torch.zeros(theta_o['module.classifier.1.bias'].size()), requires_grad=True)
#     torch.nn.init.xavier_uniform_(theta_n_bottleneck)
    theta_n = deepcopy(state_dict) # initialize all layers

    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            train_acc = 0.0
            i = 0
            running_losses = []
            acces = []
            since = time.time()
            for data in trainloader:
                step += 1
                i += 1
                # get the inputs
                inputs, labels, file = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                optimizer_s.zero_grad()
                # calculate old output and loss
#                 for n in net.state_dict():
#                     net.state_dict()[n] = theta_o[n].clone()
#                 net.load_state_dict(theta_o)
                outputs_old = net(inputs) 
                loss_old = criterion(outputs_old, labels)
                # calculate new output and update weights
                for n in theta_n.keys():
                    net.state_dict()['module.{}'.format(n)] = theta_n[n].clone()
                outputs_new = net(inputs)
                loss_new = criterion(outputs_new, labels)
                _, pred = outputs_new.max(1)
                loss_2 = loss_old + loss_new
                loss_2.backward()
                optimizer.step()
                optimizer_s.step()

                # see training acc and loss
                _, pred = outputs_new.max(1)
                num_correct = (pred == labels).sum().item()
                acc = num_correct / inputs.shape[0]
                acces.append(acc)
                running_losses.append(loss_2.data.view(1,1))

                # cache theta_new
                theta_n = deepcopy(net.state_dict())


    ##################################################################################################
    ##################################################################################################


                if step % len(trainloader) == 0:
                    valid_acc, valid_loss = valid(validloader)
                    time_elapsed = time.time() - since
                    print(
                        'epoch{}/{} time:{:.0f}m {:.0f}s train_acc:{:.3f} train_loss:{:.4f} valid_acc:{:.3f} valid loss:{:.3f}'.format(
                            epoch + 1,
                            num_epochs, time_elapsed // 60,
                            time_elapsed % 60,
                            np.mean(acces),
                            torch.cat(running_losses).mean(),
                            valid_acc,
                            valid_loss))
                    since = time.time()
    ##################################################################################################
    ######################################debug###########################################################
                    since = time.time()
                    valid_losses = []
                    valid_acces = []
                    for valid_task in range(len(validloader_previous)):
                        #             print('[Valid in task{}]:'.format(valid_task + 1))
                        valid_acc, valid_loss = valid(valid_loader_list[valid_task])
                        valid_losses.append(valid_loss.view(1,1))
                        valid_acces.append(valid_acc)
                    valid_losses=torch.cat(valid_losses)
                    time_elapsed = time.time() - since
                    print('[Validation previous task:] time:{:.0f}m {:.0f}s acc: {}, loss: {}'
                          .format(time_elapsed // 60, time_elapsed % 60, valid_acces, valid_losses.mean()))

    ##################################################################################################
    ##################################################################################################

    #         time_elapsed = time.time() - since
    #         since = time.time()
    #         valid_acc, valid_loss = valid(validloader)
    #         running_losses = torch.cat(running_losses)
    #         print(
    #             'epoch{}/{} time:{:.0f}m {:.0f}s train_acc:{:.3f} train_loss:{:.4f} valid_acc:{:.3f} valid loss:{:.3f}'.format(
    #                 epoch + 1,
    #                 num_epochs, time_elapsed // 60,
    #                 time_elapsed % 60,
    #                 sum(acces) / len(trainloader),
    #                 running_losses.mean(),
    #                 valid_acc,
    #                 valid_loss))


    return acces, running_losses


#%%
################################## here is main script for training ####################################
# define stuff related to training
num_epochs = 50


for train_task in tnrange(12):
    if train_task == 0:
        print('[Train in task{}]:'.format(train_task + 1))
        feed(train_loader_list[train_task],
             is_training=True, init_task=True, num_epochs=num_epochs,
             validloader=valid_loader_list[train_task],
             is_valid=False, is_save_csv=False, csv_name=None)
#         # see all batch validation
#         valid_losses = []
#         valid_acces = []
#         for valid_task in range(12):
#             # print('[Valid in task{}]:'.format(valid_task + 1))
#             valid_acc, valid_loss = valid(valid_loader_list[valid_task])
#             valid_losses.append(valid_loss.view(1,1))
#             valid_acces.append(valid_acc)
#         valid_losses = torch.cat(valid_losses)
#         print('[Validation all task:], acc: {}'.format(np.mean(valid_acces)))
        
    else:
        print('[Train in task{}]:'.format(train_task + 1))
        # calculate old loss and acc for previous task
        valid_losses = []
        valid_acces = []
#         for valid_task in range(train_task):
#             valid_acc, valid_loss = valid(valid_loader_list[valid_task])
#             valid_losses.append(valid_loss.view(1,1))
#             valid_acces.append(valid_acc)
#         valid_losses = torch.cat(valid_losses)
#         print('[Validation acc for previous all tasks: {}]'.format(
#             np.mean(valid_acces)))
#         print('[Validation acc for each previous tasks: {}]'.format(valid_acces))
        feed(train_loader_list[train_task],
             is_training=True, init_task=False,
             num_epochs=num_epochs,
             validloader=valid_loader_list[train_task],
             is_valid=False, is_save_csv=False, csv_name=None, validloader_previous=valid_loader_list[0:train_task])
        # see all batch validation
#         valid_losses = []
#         valid_acces = []
#         for valid_task in range(12):
#             #             print('[Valid in task{}]:'.format(valid_task + 1))
#             valid_acc, valid_loss = valid(valid_loader_list[valid_task])
#             valid_losses.append(valid_loss.view(1,1))
#             valid_acces.append(valid_acc)
#         valid_losses = torch.cat(valid_losses)
#         print('[Validation all task:], acc: {}'.format(np.mean(valid_acces
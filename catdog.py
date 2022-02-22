import torch
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms, utils
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
import cv2, sys
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import argparse
from sklearn.model_selection import train_test_split
import yaml

class ImageDataset(Dataset):
  def __init__(self,csv,img_folder,transform):
    self.csv=csv
    self.transform=transform
    self.img_folder=img_folder
     
    self.image_names=self.csv[:]['fileID']
    self.labels=np.array(self.csv[:]['labels'])

  def __len__(self):
    return len(self.image_names)
 
  def __getitem__(self,index):
    image=cv2.imread(self.img_folder+"/"+self.image_names.iloc[index])
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image=self.transform(image)
    targets=self.labels[index]
     
    sample = {'image': image,'labels':targets}
 
    return sample

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv0 = nn.Conv2d(3, 16, 3, padding=1) #Channels in, Channels out, Filter size     
        
        self.conv1 = nn.Conv2d(16, 32, 3, padding=1) 
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.25)
        
        # fully conected layers:
        self.fc15 = nn.Linear(12544, 4096)
        self.fc16 = nn.Linear(4096, 512)
        self.fc17 = nn.Linear(512, 2)

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = self.dropout(x)
        x= self.pool(x)

        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x= self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x= self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool(x)
             
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.pool(x)
        
        x = x.reshape(-1, 12544)
        x = F.relu(self.fc15(x))
        x = self.dropout(x)
        x = F.relu(self.fc16(x))
        x = self.fc17(x)
        return x

train_transforms =  transforms.Compose([
      transforms.ToPILImage(), #Transforms only works on PIL images (not numpy arrays)
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
    ])

test_transforms = transforms.Compose([   
      transforms.ToPILImage(),
      transforms.Resize((224, 224)),
      transforms.ToTensor()
    ])

val_transforms = transforms.Compose([   
      transforms.ToPILImage(),
      transforms.Resize((224, 224)),
      transforms.ToTensor()
    ])

def labeled_dataset_df(train_dir):
    filenames=[]
    labels=[]

    for root, dir, files in os.walk(train_dir):
        for filename in files:
            if filename[:3]=="cat":
                filenames.append(filename)
                labels.append(0)
            else:
                filenames.append(filename)
                labels.append(1)

    data_dict = {'fileID': filenames, 'labels': labels}
    df = pd.DataFrame(data=data_dict)
    return df

def read_config(config_file):
    with open('config.yaml') as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

def create_dataloaders(config):
    df = labeled_dataset_df(config['train_dir'])
    
    train_list, val_list = train_test_split(df, test_size=config['val_size'])
    
    train_dataset=ImageDataset(train_list, config['train_dir'], train_transforms)
    val_dataset=ImageDataset(val_list, config['train_dir'], val_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    return train_dataloader, val_dataloader

def train(train_dataloader, val_dataloader, config):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    for epoch in range(config['epoch']):  

        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a dict {'image', 'labels'}
            inputs = data['image'] 
            labels = data['labels']

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 1:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2:.3f}')
                running_loss = 0.0

    print('Finished Training')
    return net
    
    
#read yaml file
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    
    config = read_config(sys.argv[1])
    train_dataloader, val_dataloader = create_dataloaders(config)
    model = train(train_dataloader, val_dataloader, config)
    print(model)
    



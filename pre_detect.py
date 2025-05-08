import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from osgeo import gdal
import torch.nn.functional as F



#img data =np.stack((c09,c10,c11,c12,c13, c14,c15,#7bt1, bt2, bt3,#3 saz,dem,lat,lon, #4T500,T700,T850,T925,R500,R700,R850,R925,# 8div500,div700 div850.div925.#4k1, k2, k3,TP), axis=0)

class CustomDataset(Dataset):
    def __init__(self, mem_paths, input_height, input_width):
        self.mem_paths = mem_paths
        #self.segs_paths = segs_paths
        self.input_height = input_height
        self.input_width = input_width

    def __len__(self):
        return len(self.mem_paths)

    def __getitem__(self, idx):
        mem_file = self.mem_paths[idx]


        try:
            dataset = gdal.Open(mem_file, gdal.GA_ReadOnly)
            bands = [np.array(dataset.GetRasterBand(i).ReadAsArray()) for i in range(1, 28)]

            b1, b2, b3, b4, b5, b6, b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27 = bands

      
            input_data = np.stack([b1, b2, b3, b4, b5, b6, b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27], axis=0)
            input_data[input_data<0]=0
            return input_data
        except Exception as e:
            print(f"Error processing files {mem_file}: {e}")


class UNet(nn.Module):
    def __init__(self, num_channels):
        super(UNet, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(num_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2)

        # Middle layer
        self.conv_mid1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv_mid2 = nn.Conv2d(512, 512, 3, padding=1)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.conv8 = nn.Conv2d(256, 256, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv9 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv10 = nn.Conv2d(128, 128, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv11 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)

        # Output layer
        self.conv_out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        conv1 = torch.relu(self.conv2(conv1))
        pool1 = self.pool1(conv1)

        conv3 = self.conv3(pool1)
        conv3 = torch.relu(self.conv4(conv3))
        pool2 = self.pool2(conv3)

        conv5 = self.conv5(pool2)
        conv5 = torch.relu(self.conv6(conv5))
        pool3 = self.pool3(conv5)

        # Middle layer
        conv_mid = self.conv_mid1(pool3)
        conv_mid = torch.relu(self.conv_mid2(conv_mid))

        # Decoder
        up3 = self.up3(conv_mid)
        up3 = torch.cat([conv5, up3], dim=1)
        conv7 = self.conv7(up3)
        conv7 = torch.relu(self.conv8(conv7))

        up2 = self.up2(conv7)
        up2 = torch.cat([conv3, up2], dim=1)
        conv9 = self.conv9(up2)
        conv9 = torch.relu(self.conv10(conv9))

        up1 = self.up1(conv9)
        up1 = torch.cat([conv1, up1], dim=1)
        conv11 = self.conv11(up1)
        conv11 = torch.relu(self.conv12(conv11))

        # Output layer
        output = torch.sigmoid(self.conv_out(conv11))
        return output.squeeze(1)


def load_and_preprocess_data(mem_path, input_height, input_width, batch_size=16, if_eval=False):
    mem = []
    segmentations = []

    mem_train = sorted(mem)[:int(1.0 * len(mem))]

    
    dataset_train = CustomDataset(mem_train, input_height, input_width)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return loader_train



if __name__ == '__main__':
    
  
    input_height = 32
    input_width = 32
    num_channels = 27

    nv=41209

    model = UNet(num_channels)

    checkpoint = torch.load('unet.pt', map_location=torch.device('cpu'))  
    
    model.load_state_dict(checkpoint)

    
    data_generator_val=load_and_preprocess_data("./input27/", input_height, input_width)


    vpredictions_array = []
    vlabels_array = []
    tpredictions_array = []
    tlabels_array = []


    model.eval()
    with torch.no_grad():
            for X_val in data_generator_val:
                
                val_outputs = model(X_val)

                vpredictions_array.append(val_outputs)


    vpredictions_array = np.concatenate(vpredictions_array)


    vp_reshaped = vpredictions_array.reshape(nv, 1024)
    np.savetxt('./p.txt', vp_reshaped)  




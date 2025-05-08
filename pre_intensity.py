import numpy as np
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
import torch.nn.functional as F
import os
import torchvision.models as models

backbone = 'resnet34'

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
        #seg_file = self.segs_paths[idx]

        try:
            dataset = gdal.Open(mem_file, gdal.GA_ReadOnly)
            bands = [np.array(dataset.GetRasterBand(i).ReadAsArray()) for i in range(1, 37)]

            b1,b2,b3,b4,b5,b6,b7,b8, b9, b10, b11, b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33= bands
  
            input_data = np.stack([b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33], axis=0)
            input_data = np.clip(input_data, 0, 1)
            return input_data
        except Exception as e:
            print(f"Error processing files {mem_file} : {e}")


def load_and_preprocess_data(mem_path, input_height, input_width, batch_size=16, if_eval=False):
    mem = []
    segmentations = []
    for root, _, files in os.walk(mem_path):
        for file in files:
            if file.endswith('.tif'):
                mem.append(os.path.join(root, file))



    mem_train = sorted(mem)[:int(1.0 * len(mem))]
  

    dataset_train = CustomDataset(mem_train, input_height, input_width)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    return loader_train

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, upsample_mode='pixelshuffle', BN_enable=True):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.upsample_mode = upsample_mode
        self.BN_enable = BN_enable

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1,
                              bias=False)

        if self.BN_enable:
            self.norm1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

        if self.upsample_mode == 'deconv':
            self.upsample = nn.ConvTranspose2d(in_channels=mid_channels, out_channels=out_channels,
                                               kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        elif self.upsample_mode == 'pixelshuffle':
            self.upsample = nn.PixelShuffle(upscale_factor=2)
        if self.BN_enable:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.BN_enable:
            x = self.norm1(x)
        x = self.relu1(x)
        x = self.upsample(x)
        if self.BN_enable:
            x = self.norm2(x)
        x = self.relu2(x)
        return x


class Resnet_Unet(nn.Module):

    def __init__(self, inputchannel, outputchannel, BN_enable=True, resnet_pretrain=False):
        super().__init__()
        self.BN_enable = BN_enable
 
        if backbone == 'resnet34':
            resnet = models.resnet34(pretrained=resnet_pretrain)
            filters = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=resnet_pretrain)
            filters = [64, 256, 512, 1024, 2048]
        self.firstconv = nn.Conv2d(in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder
        self.center = DecoderBlock(in_channels=filters[3], mid_channels=filters[3] * 4, out_channels=filters[3],
                                   BN_enable=self.BN_enable)
        self.decoder1 = DecoderBlock(in_channels=filters[3] + filters[2], mid_channels=filters[2] * 4,
                                     out_channels=filters[2], BN_enable=self.BN_enable)
        self.decoder2 = DecoderBlock(in_channels=filters[2] + filters[1], mid_channels=filters[1] * 4,
                                     out_channels=filters[1], BN_enable=self.BN_enable)
        self.decoder3 = DecoderBlock(in_channels=filters[1] + filters[0], mid_channels=filters[0] * 4,
                                     out_channels=filters[0], BN_enable=self.BN_enable)
        if self.BN_enable:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=outputchannel, kernel_size=1),
                nn.Sigmoid()
            )
        else:
            self.final = nn.Sequential(
                nn.Conv2d(in_channels=filters[0], out_channels=32, kernel_size=3, padding=1),
                nn.ReLU(inplace=False),
                nn.Conv2d(in_channels=32, out_channels=outputchannel, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.encoder1(x_)  
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)

        center = self.center(e3)

        d2 = self.decoder1(torch.cat([center, e2], dim=1))
        d3 = self.decoder2(torch.cat([d2, e1], dim=1))
        d4 = self.decoder3(torch.cat([d3, x], dim=1))

        return self.final(d4)
if __name__ == '__main__':

    input_height = 32
    input_width = 32
    num_channels = 33

    nv=41209

    #model = UNet(num_channels)
    model = Resnet_Unet(num_channels,1, BN_enable=True, resnet_pretrain=True)

    checkpoint = torch.load('resunet.pt', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)


    data_generator_val=load_and_preprocess_data("./input33/", input_height, input_width)
 
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
    np.savetxt('./pre.txt', vp_reshaped) 
 


import sys
sys.path.append('.')
import torch
import cv2
from torchvision.datasets.vision import VisionDataset
from pathlib import Path
import numpy as np 
from model.od.data.datasets import letterbox
from typing import Any

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class veclocityDataset(Dataset):
    def __init__(self, root, image_size=480):
        self.root = root
        self.frames_path = os.path.join(root, 'Frames')
        self.pixelwise_depths_path = os.path.join(root, 'Pixelwise_Depths')
        self.poses_path = os.path.join(root, 'Poses', 'colon_position_rotation.csv')
        self.poses_data = pd.read_csv(self.poses_path)
        self.image_size = image_size
        # Get the list of image files
        self.image_files = sorted([file for file in os.listdir(self.frames_path) if file.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Read RGB images
        image1_path = os.path.join(self.frames_path, self.image_files[idx])
        image2_path = os.path.join(self.frames_path, self.image_files[idx + 1])
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)


        # Read depth images
        depth1_path = os.path.join(self.pixelwise_depths_path, 'aov_' + self.image_files[idx])
        depth2_path = os.path.join(self.pixelwise_depths_path, 'aov_' + self.image_files[idx + 1])
        depth1 = Image.open(depth1_path)
        depth2 = Image.open(depth2_path)

        # Get camera positions
        pose1 = self.poses_data.iloc[idx]
        pose2 = self.poses_data.iloc[idx + 1]
        
        # Convert camera positions to tensors
        position1 = torch.tensor([pose1['tX'], pose1['tY'], pose1['tZ'], pose1['rX'], pose1['rY'], pose1['rZ']]).float()
        position2 = torch.tensor([pose2['tX'], pose2['tY'], pose2['tZ'], pose2['rX'], pose2['rY'], pose2['rZ']]).float()

        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
            ])
        
        image1 = transform(image1)
        image2 = transform(image2)
        depth1 = transform(depth1)
        depth2 = transform(depth2)
        

        return image1, image2, depth1, depth2, position1, position2
import json
import os
import pickle
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# import utils.data_transforms as transforms
from utils.data_transforms import Normalize, ToTensor

from dataset.real_depth_utils import depth_map_to_voxel
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated


class DemoDataset(Dataset):

    def calculated_ray_direction(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.camera_model.camera2world_ray(points)
        return ray

    def __init__(self, config, img_dir, depth_dir, voxel_output=False, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        self.img_dir = img_dir
        self.depth_dir = depth_dir
        self.voxel_output = voxel_output

        self.normalize = Normalize(mean=img_mean, std=img_std)
        self.to_tensor = ToTensor()

        self.img_size = config.image_shape

        self.camera_model_path = config.dataset.camera_calibration_path
        self.camera_model = FishEyeCameraCalibrated(calibration_file_path=self.camera_model_path)

        self.ray = self.calculated_ray_direction(config.dataset.image_width, config.dataset.image_height)

        self.data_list = self.get_input_data(self.img_dir, self.depth_dir)

        self.cuboid_side = config.model.cuboid_side
        self.volume_size = config.model.volume_size

    def get_input_data(self, img_dir, depth_dir):
        print("start loading test file")
        data_list = []

        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            depth_path = os.path.join(depth_dir, f'{img_name}.exr')
            if not os.path.exists(depth_path):
                raise Exception(f"The depth map {depth_path} does not exist!")

            data_list.append({'img_path': img_path, 'depth_path': depth_path})
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        img_path = self.data_list[index]['img_path']
        depth_path = self.data_list[index]['depth_path']


        raw_img = cv2.imread(img_path)
        raw_img = raw_img[:, 128: -128, :]

        # data augmentation
        img = cv2.resize(raw_img, dsize=(256, 256)) / 255.

        img_rgb = img[:, :, ::-1]
        img_rgb = np.ascontiguousarray(img_rgb)

        img_torch = self.normalize(img)
        img_torch = self.to_tensor(img_torch)
        img_rgb = self.normalize(img_rgb)
        img_rgb_torch = self.to_tensor(img_rgb)

        depth_map = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_map.shape[0] != 1024 or depth_map.shape[1] != 1280:
            depth_map = cv2.resize(depth_map, (1280, 1024), interpolation=cv2.INTER_NEAREST)
        if len(depth_map.shape) == 3:
            depth_map = depth_map[:, :, 0]
        depth_map[depth_map > 10] = 10

        if self.voxel_output is True:
            depth_scene_info = depth_map_to_voxel(self.ray, depth_map, self.cuboid_side, self.volume_size)
        else:
            depth_scene_info = torch.from_numpy(depth_map).float()

        return img_torch, img_rgb_torch, depth_scene_info, img_path

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

from utils.calculate_errors import align_skeleton, calculate_error
from dataset.real_depth_utils import depth_map_to_voxel
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated


class TestDataset(Dataset):
    path_dict = {
        'jian1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/05082022/jian1',
        },
        'new_jian1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian1',
        },
        'new_jian2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/jian2',
        },
        'new_diogo1': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo1',
        },
        'new_diogo2': {
            'path': r'/HPS/ScanNet/work/egocentric_view/25082022/diogo2',
        },

    }

    def calculated_ray_direction(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.camera_model.camera2world_ray(points)
        return ray

    def __init__(self, config, seq_name, estimated_depth_name=None, voxel_output=True,
                 local_machine=False, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225), with_one_depth=False):
        self.seq_name = seq_name
        self.voxel_output = voxel_output
        self.estimated_depth_name = estimated_depth_name
        self.with_one_depth = with_one_depth # only for visualization

        self.normalize = Normalize(mean=img_mean, std=img_std)
        self.to_tensor = ToTensor()

        self.local_machine = local_machine
        self.img_size = config.image_shape

        self.camera_model_path = config.dataset.camera_calibration_path
        self.camera_model = FishEyeCameraCalibrated(calibration_file_path=self.camera_model_path)

        self.ray = self.calculated_ray_direction(config.dataset.image_width, config.dataset.image_height)

        self.image_path_list, self.gt_pose_list, self.depth_map_list = self.get_gt_data(seq_name)

        assert len(self.image_path_list) == len(self.gt_pose_list) and len(self.gt_pose_list) == len(
            self.depth_map_list)

        self.cuboid_side = config.model.cuboid_side
        self.volume_size = config.model.volume_size

    def get_gt_data(self, seq_name):
        print("start loading test file")
        base_path = self.path_dict[seq_name]['path']
        if self.local_machine:
            base_path = base_path.replace('/HPS', 'X:').replace('/CT', 'Z:')

        img_data_path = os.path.join(base_path, 'imgs')
        gt_path = os.path.join(base_path, 'local_pose_gt.pkl')
        if self.estimated_depth_name is not None:
            depth_path = os.path.join(base_path, self.estimated_depth_name)
        else:
            depth_path = os.path.join(base_path, 'rendered', 'depths')
        syn_path = os.path.join(base_path, 'syn.json')

        with open(syn_path, 'r') as f:
            syn_data = json.load(f)

        ego_start_frame = syn_data['ego']
        ext_start_frame = syn_data['ext']

        with open(gt_path, 'rb') as f:
            pose_gt_data = pickle.load(f)

        image_path_list = []
        gt_pose_list = []
        depth_path_list = []

        for pose_gt_item in pose_gt_data:
            ext_id = pose_gt_item['ext_id']
            ego_pose_gt = pose_gt_item['ego_pose_gt']
            if ego_pose_gt is None:
                continue
            ego_id = ext_id - ext_start_frame + ego_start_frame
            egocentric_image_name = "img_%06d.jpg" % ego_id
            depth_name = "img_%06d" % ego_id

            image_path = os.path.join(img_data_path, egocentric_image_name)
            if not os.path.exists(image_path):
                continue
            image_path_list.append(image_path)
            if self.estimated_depth_name is not None:
                depth_full_path = os.path.join(depth_path, 'img_%06d.jpg.exr' % ego_id)
            else:
                depth_full_path = os.path.join(depth_path, depth_name, 'Image0001.exr')
            depth_path_list.append(depth_full_path)
            gt_pose_list.append(ego_pose_gt)
        print("dataset length: {}".format(len(image_path_list)))
        return image_path_list, gt_pose_list, depth_path_list

    def evaluate_mpjpe(self, predicted_pose_list):
        gt_pose_list = self.gt_pose_list

        mpjpe = calculate_error(predicted_pose_list, gt_pose_list)

        # align the estimated result and original result
        aligned_estimated_result, gt_seq = align_skeleton(predicted_pose_list, gt_pose_list, None)

        pampjpe = calculate_error(aligned_estimated_result, gt_seq)

        return mpjpe, pampjpe

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        img_path = self.image_path_list[index]
        if self.with_one_depth:
            # only for visualization
            depth_path = self.depth_map_list[2070]
        else:
            depth_path = self.depth_map_list[index]

        if self.local_machine:
            img_path = img_path.replace('/HPS', 'X:').replace('/CT', 'Z:')
            depth_path = depth_path.replace('/HPS', 'X:').replace('/CT', 'Z:')

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

def main():
    dataset = TestDataset(config=None, seq_name='new_jian1',
                          voxel_output=False, local_machine=True)

    img_torch, img_rgb_torch, depth_scene_info, img_path = dataset[100]


if __name__ == '__main__':
    main()
from copy import deepcopy
import numpy as np
import pickle
import random
import time
from copy import copy
import cv2

# from utils_proj.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
# from utils_proj.fisheye.FishEyeEquisolid import FishEyeCameraEquisolid

import torch
from torch import nn

from utils import op, multiview, img, misc, volumetric
from torch.nn.functional import interpolate

from network import pose_resnet
from network.v2v import V2VModel
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated


class VoxelNetwork_depth(nn.Module):
    def __init__(self, config, device='cuda'):
        super(VoxelNetwork_depth, self).__init__()

        self.device = device
        self.num_joints = config.model.backbone.num_joints

        # volume
        self.volume_softmax = config.model.volume_softmax
        self.volume_multiplier = config.model.volume_multiplier
        self.volume_size = config.model.volume_size

        self.cuboid_side = config.model.cuboid_side

        self.kind = config.model.kind

        # heatmap
        self.heatmap_softmax = config.model.heatmap_softmax
        self.heatmap_multiplier = config.model.heatmap_multiplier

        # # transfer
        # self.transfer_cmu_to_human36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

        network_path = config.model.backbone.checkpoint
        network_loads = torch.load(network_path)

        self.backbone = pose_resnet.get_pose_net(state_dict=network_loads['state_dict'])
        print("warning!!! Use default backbone resnet")
        self.backbone = self.backbone.to(device)

        if config.opt.train_2d is False:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # resize and pad feature for reprojection
        self.process_features = nn.Sequential(
            nn.Conv2d(256, 32, 1),
            nn.Upsample(size=(1024, 1024)),
            nn.ConstantPad2d(padding=(128, 128, 0, 0), value=0.0)
        )
        self.process_features = self.process_features.to(device)

        self.with_scene = config.model.with_scene
        if config.model.with_scene is True:
            if config.model.with_intersection is True:
                volume_input_channel_num = 32 + 1 + 32
                self.with_intersection = True
            else:
                volume_input_channel_num = 32 + 1
                self.with_intersection = False
        else:
            volume_input_channel_num = 32

        self.volume_net = V2VModel(volume_input_channel_num, self.num_joints)
        self.volume_net = self.volume_net.to(device)

        print('build coord volume')
        self.coord_volume = self.build_coord_volume()
        self.coord_volumes = self.coord_volume.unsqueeze(0).expand(config.opt.batch_size,
                                                                   -1, -1, -1, -1)
        self.coord_volumes = self.coord_volumes.to(device)

        self.fisheye_camera_model = FishEyeCameraCalibrated(
            calibration_file_path=config.dataset.camera_calibration_path)
        print('build reprojected grid coord')
        self.grid_coord_proj = op.get_projected_2d_points_with_coord_volumes(fisheye_model=self.fisheye_camera_model,
                                                                             coord_volume=self.coord_volume)
        self.grid_coord_proj.requires_grad = False

        self.grid_coord_proj_batch = op.get_grid_coord_proj_batch(self.grid_coord_proj,
                                                                  batch_size=config.opt.batch_size,
                                                                  heatmap_shape=config.heatmap_shape)
        self.grid_coord_proj_batch.requires_grad = False
        self.grid_coord_proj_batch = self.grid_coord_proj_batch.to(device)

        # self.fisheye_camera_model_depth = FishEyeCameraEquisolid(focal_length=9, sensor_size=32, img_size=(1280, 1024))

        # self.ray_torch = self.calculated_ray_direction(config.dataset.image_width,
        #                                                config.dataset.image_height).to(device)
        # self.ray_torch.requires_grad = False

        self.ray = self.calculated_ray_direction_numpy(config.dataset.image_width,
                                                       config.dataset.image_height)

        self.image_width = config.dataset.image_width
        self.image_height = config.dataset.image_height

    def build_coord_volume(self):
        """
        get coord volume and prepare for the re-projection process
        :param self:
        :return:
        """
        # build coord volumes
        sides = np.array([self.cuboid_side, self.cuboid_side, self.cuboid_side])

        position = np.array([-self.cuboid_side / 2, -self.cuboid_side / 2, 0])
        # build coord volume
        xxx, yyy, zzz = torch.meshgrid(torch.arange(self.volume_size),
                                       torch.arange(self.volume_size),
                                       torch.arange(self.volume_size))
        grid = torch.stack([xxx, yyy, zzz], dim=-1).type(torch.float)
        grid = grid.reshape((-1, 3))

        grid_coord = torch.zeros_like(grid)
        grid_coord[:, 0] = position[0] + (sides[0] / (self.volume_size - 1)) * grid[:, 0]
        grid_coord[:, 1] = position[1] + (sides[1] / (self.volume_size - 1)) * grid[:, 1]
        grid_coord[:, 2] = position[2] + (sides[2] / (self.volume_size - 1)) * grid[:, 2]

        coord_volume = grid_coord.reshape(self.volume_size, self.volume_size, self.volume_size, 3)

        return coord_volume

    def calculated_ray_direction(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.fisheye_camera_model.camera2world_ray(points)
        ray_torch = torch.from_numpy(ray)
        return ray_torch

    def calculated_ray_direction_numpy(self, image_width, image_height):
        points = np.zeros(shape=(image_width, image_height, 2))
        x_range = np.array(range(image_width))
        y_range = np.array(range(image_height))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        ray = self.fisheye_camera_model.camera2world_ray(points)
        return ray

    def depth_to_voxel_pytorch(self, depth):
        # directly multiply the depth on the pre-calculated rays
        # resize depth to (image_width, image_height)
        depth = depth.unsqueeze(0)
        depth = interpolate(depth, (self.image_height, self.image_height), mode='nearest')
        depth = torch.nn.functional.pad(depth, pad=[128, 128])
        depth = depth.squeeze(0)[0]
        print(depth.shape)
        depth_img_flat = depth.transpose(0, 1).reshape(-1)
        point_cloud = self.ray_torch.transpose(0, 1) * depth_img_flat
        point_cloud = point_cloud.transpose(0, 1)

        # point cloud to voxel

        voxel_torch = self.point_cloud_to_voxel_pytorch(point_cloud)
        return voxel_torch

    def point_cloud_to_voxel_pytorch(self, point_cloud):
        scene_point_cloud_local = point_cloud

        scene_point_cloud_local[:, 0] = (scene_point_cloud_local[:,
                                         0] + self.cuboid_side / 2) * self.volume_size / self.cuboid_side
        scene_point_cloud_local[:, 1] = (scene_point_cloud_local[:,
                                         1] + self.cuboid_side / 2) * self.volume_size / self.cuboid_side
        scene_point_cloud_local[:, 2] = (scene_point_cloud_local[:, 2]) * self.volume_size / self.cuboid_side

        scene_point_cloud_local = torch.round(scene_point_cloud_local).long()

        # note: the gradient is zero here!!!
        good_indices = torch.logical_and(self.volume_size-1 >= scene_point_cloud_local, scene_point_cloud_local >= 0)
        good_indices = torch.all(good_indices, dim=1)
        scene_point_cloud_local = scene_point_cloud_local[good_indices]
        # scene_point_cloud_local = np.clip(scene_point_cloud_local, a_min=0, a_max=self.volume_size - 1).astype(np.int)
        voxel_torch = torch.zeros(size=(self.volume_size, self.volume_size, self.volume_size)).to(self.device)
        voxel_torch[scene_point_cloud_local.cpu().numpy().T] = 1
        return voxel_torch

    def depth_map_to_voxel_numpy(self, depth):
        # directly multiply the depth on the pre-calculated rays
        depth = depth.view(depth.size(-2), depth.size(-1)).cpu().detach().numpy()
        depth = cv2.resize(depth, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
        depth = np.pad(depth, ((0, 0), (128, 128)), 'constant', constant_values=0)
        depth_img_flat = depth.T.reshape((-1))
        point_cloud = self.ray.T * depth_img_flat
        point_cloud = point_cloud.T

        # point cloud to voxel
        voxel_torch = self.point_cloud_to_voxel_numpy(point_cloud)
        return voxel_torch

    def point_cloud_to_voxel_numpy(self, point_cloud):
        scene_point_cloud_local = copy(point_cloud)
        scene_point_cloud_local[:, 0] = (scene_point_cloud_local[:,
                                         0] + self.cuboid_side / 2) * self.volume_size / self.cuboid_side
        scene_point_cloud_local[:, 1] = (scene_point_cloud_local[:,
                                         1] + self.cuboid_side / 2) * self.volume_size / self.cuboid_side
        scene_point_cloud_local[:, 2] = (scene_point_cloud_local[:, 2]) * self.volume_size / self.cuboid_side

        scene_point_cloud_local = np.round_(scene_point_cloud_local)
        good_indices = np.logical_and(self.volume_size-1 >= scene_point_cloud_local, scene_point_cloud_local >= 0)
        good_indices = np.all(good_indices, axis=1)
        scene_point_cloud_local = scene_point_cloud_local[good_indices]
        # scene_point_cloud_local = np.clip(scene_point_cloud_local, a_min=0, a_max=self.volume_size - 1).astype(np.int)
        voxel_torch = torch.zeros(size=(self.volume_size, self.volume_size, self.volume_size)).to(self.device)
        voxel_torch[scene_point_cloud_local.T] = 1
        return voxel_torch

    def forward(self, images, grid_coord_proj_batch, coord_volumes, scene_volumes=None, depth_map_batch=None):
        """
        side: the length of volume square side, like 2 meters or 3 meters
        volume_size: the number of grid, like we have 64 or 32 grids for each side
        :param images:
        :return:
        """
        device = images.device
        batch_size = images.shape[0]

        # forward backbone
        heatmaps, features = self.backbone(images)

        # process features before unprojecting
        features = self.process_features(features)

        # lift to volume
        if features.shape[0] < grid_coord_proj_batch.shape[0]:
            grid_coord_proj_batch = grid_coord_proj_batch[:features.shape[0]]
        volumes = op.unproject_heatmaps_one_view_batch(features, grid_coord_proj_batch, self.volume_size)

        if self.with_scene is True:
            if scene_volumes is not None:
                # combine scene volume with project pose volume
                scene_volumes = torch.unsqueeze(scene_volumes, dim=1)
                volumes = torch.cat([volumes, scene_volumes], dim=1)
            elif depth_map_batch is not None:
                voxel_list = []
                for depth_map in depth_map_batch:
                    voxel_torch = self.depth_map_to_voxel_numpy(depth_map)
                    # show_voxel_torch(voxel_torch)
                    voxel_list.append(voxel_torch)
                scene_volumes = torch.stack(voxel_list, dim=0)
                scene_volumes = torch.unsqueeze(scene_volumes, dim=1)
                if self.with_intersection is True:
                    intersection = volumes * scene_volumes
                    volumes = torch.cat([volumes, intersection, scene_volumes], dim=1)
                else:
                    volumes = torch.cat([volumes, scene_volumes], dim=1)
            else:
                print("no scene volume or depth input!")
                return None

        # integral 3d
        volumes = self.volume_net(volumes)
        if volumes.shape[0] < coord_volumes.shape[0]:
            coord_volumes = coord_volumes[:volumes.shape[0]]
        vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes * self.volume_multiplier,
                                                                            coord_volumes,
                                                                            softmax=self.volume_softmax)

        return vol_keypoints_3d, features, volumes, self.coord_volumes


def run_voxel_net():
    image_batch = torch.ones(size=(4, 3, 256, 256))
    from utils import cfg
    config_path = 'experiments/mo2cap2/train/mo2cap2_vol_softmax.yaml'
    config = cfg.load_config(config_path)
    projection_network = VoxelNetwork_depth(config=config, device='cuda')

    vol_keypoints_3d, features, volumes, coord_volumes = projection_network(image_batch)
    print(vol_keypoints_3d.shape)


def show_voxel_torch(voxel):
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    # show voxel result
    voxel_np = voxel.cpu().numpy()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    ax.voxels(voxel_np, edgecolor="k")
    plt.savefig('tmp/0.png')
    exit(0)
    # plt.show()

def calculate_max_position():
    volumes = torch.ones(size=(4, 15, 64, 64, 64)) * 0

    volumes[:, :, 32, 32, 32] = 1
    volumes[:, :, 31, 31, 31] = 1

    from utils import cfg
    config_path = '../experiments/mo2cap2/train/mo2cap2_vol_softmax.yaml'
    config = cfg.load_config(config_path)
    projection_network = VoxelNetwork_depth(config=config, device='cpu')

    coord_volumes = projection_network.coord_volumes

    vol_keypoints_3d, volumes = op.integrate_tensor_3d_with_coordinates(volumes,
                                                                        coord_volumes,
                                                                        softmax=True)

    print(vol_keypoints_3d)
    print(vol_keypoints_3d.shape)


def visualize_grid_coord_proj():
    from utils import cfg
    from visualization.visualization_projected_grid import draw_points
    import cv2
    config_path = 'experiments/mo2cap2/train/mo2cap2_vol_softmax.yaml'
    config = cfg.load_config(config_path)
    projection_network = VoxelNetwork_depth(config=config, device='cpu')

    grid_coord_proj = projection_network.grid_coord_proj

    canvas = cv2.imread(
        r'\\winfs-inf\HPS\Mo2Cap2Plus1\static00\EgocentricData\old_data\kitchen_2\imgs\img_-04032020185303-3.jpg')

    canvas = draw_points(canvas, grid_coord_proj)

    cv2.imshow('img', canvas)
    cv2.waitKey(0)


def test_reproject_feature():
    from visualization.visualization_3D_grid import visualize_3D_grid_single_grid
    from utils import cfg
    import cv2
    config_path = 'experiments/local/train/mo2cap2_vol_softmax.yaml'
    config = cfg.load_config(config_path)
    projection_network = VoxelNetwork_depth(config=config, device='cpu')

    # features = torch.zeros(size=(4, 15, 1024, 1280))
    #
    # features[:, :, 490:570, 620:700] = 10

    # read feature from egocentric map

    heatmap_path = r'X:\Mo2Cap2Plus1\static00\EgocentricData\REC23102020\studio-jian1\da_external_layer4\img-10232020185916-700.mat'
    from scipy.io import loadmat
    heatmap = loadmat(heatmap_path)
    heatmap = heatmap['heatmap']
    # recover heatmap
    heatmap = cv2.resize(heatmap, dsize=(1024, 1024), interpolation=cv2.INTER_NEAREST)
    heatmap = np.pad(heatmap, ((0, 0), (128, 128), (0, 0)), mode='edge')
    heatmap = heatmap.transpose((2, 0, 1))

    heatmap = np.expand_dims(heatmap, axis=0)

    features = torch.from_numpy(heatmap)

    grid_coord_proj_batch = op.get_grid_coord_proj_batch(projection_network.grid_coord_proj,
                                                         batch_size=features.shape[0],
                                                         heatmap_shape=(1024, 1280),
                                                         device='cpu')

    volumes_batch = op.unproject_heatmaps_one_view_batch(features, grid_coord_proj_batch,
                                                         projection_network.volume_size)

    volumes = op.unproject_heatmaps_one_view(features, projection_network.grid_coord_proj,
                                             projection_network.volume_size)

    print(torch.sum(torch.abs(volumes - volumes_batch)))
    # # visualize volume
    # print(volumes.shape)
    # for i in range(15):
    #     volume_single = volumes[0][i]
    #
    #     zoomed_volume_single = zoom(volume_single, 0.5) * 100
    #     point_cloud = visualize_3D_grid_single_grid(zoomed_volume_single)
    #     coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
    #     open3d.visualization.draw_geometries([point_cloud, coord])


if __name__ == '__main__':
    # visualize_grid_coord_proj()
    # test_reproject_feature()
    # calculate_max_position()
    run_voxel_net()

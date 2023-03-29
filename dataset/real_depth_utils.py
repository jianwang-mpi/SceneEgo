import json
import numpy as np
from copy import copy
import torch

def calcualate_depth_scale(depth_scale_json_file, log_err=False):
    with open(depth_scale_json_file, 'r') as f:
        depth_scale_data_list = json.load(f)
    # print(depth_scale_data_list)

    scale_list = []
    for scale_data in depth_scale_data_list:
        x1 = scale_data['x1']
        x2 = scale_data['x2']

        x1 = np.asarray(x1)
        x2 = np.asarray(x2)

        distance = np.linalg.norm(x2 - x1)
        # print(distance)
        scale = scale_data['real'] / distance
        scale_list.append(scale)
    if log_err:
        print(scale_list)
        print(np.std(scale_list) / np.average(scale_list))
    # print(np.average(scale_list))
    return np.average(scale_list)

def depth_map_to_voxel(ray, depth, cuboid_side, volume_size):
    # directly multiply the depth on the pre-calculated rays
    depth_img_flat = depth.T.reshape((-1))
    point_cloud = ray.T * depth_img_flat
    point_cloud = point_cloud.T

    # import open3d
    # point_cloud_vis = open3d.geometry.PointCloud()
    # point_cloud_vis.points = open3d.utility.Vector3dVector(point_cloud)
    # coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    # open3d.visualization.draw_geometries([point_cloud_vis, coord])

    # point cloud to voxel
    voxel_torch = point_cloud_to_voxel_pytorch(point_cloud, cuboid_side, volume_size)
    return voxel_torch

def point_cloud_to_voxel_pytorch(point_cloud, cuboid_side, volume_size):
    scene_point_cloud_local = copy(point_cloud)
    scene_point_cloud_local[:, 0] = (scene_point_cloud_local[:,
                                     0] + cuboid_side / 2) * volume_size / cuboid_side
    scene_point_cloud_local[:, 1] = (scene_point_cloud_local[:,
                                     1] + cuboid_side / 2) * volume_size / cuboid_side
    scene_point_cloud_local[:, 2] = (scene_point_cloud_local[:, 2]) * volume_size / cuboid_side

    scene_point_cloud_local = np.round_(scene_point_cloud_local)
    good_indices = np.logical_and(volume_size-1 >= scene_point_cloud_local, scene_point_cloud_local >= 0)
    good_indices = np.all(good_indices, axis=1)
    scene_point_cloud_local = scene_point_cloud_local[good_indices]
    # scene_point_cloud_local = np.clip(scene_point_cloud_local, a_min=0, a_max=self.volume_size - 1).astype(np.int)
    voxel_torch = torch.zeros(size=(volume_size, volume_size, volume_size))
    voxel_torch[scene_point_cloud_local.T] = 1
    return voxel_torch

if __name__ == '__main__':
    json_file = r'\\winfs-inf\CT\EgoMocap\work\EgoBodyInContext\sfm_test_data\jian3\scale.json'
    result = calcualate_depth_scale(json_file, log_err=True)
    print(result)
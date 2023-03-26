import cv2
import numpy as np
import open3d
from utils.fisheye.FishEyeCalibrated import FishEyeCameraCalibrated
from utils.fisheye.FishEyeEquisolid import FishEyeCameraEquisolid
import os


class Depth2PointCloud:
    def __init__(self, visualization, camera_model='utils/fisheye/fisheye.calibration_05_08.json',
                 post_process=True):
        if camera_model == 'FishEyeCameraEquisolid':
            self.camera = FishEyeCameraEquisolid(focal_length=9, sensor_size=32, img_size=(1280, 1024))
        else:
            self.camera = FishEyeCameraCalibrated(calibration_file_path=camera_model)
        self.visualization = visualization
        self.post_process = post_process

    def depth2pointcloud(self, camera, depth_img, real_img):
        depth_img = depth_img.transpose()
        real_img = np.transpose(real_img, axes=(1, 0, 2))
        points = np.zeros(shape=(depth_img.shape[0], depth_img.shape[1], 2))
        x_range = np.array(range(depth_img.shape[0]))
        y_range = np.array(range(depth_img.shape[1]))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        depth_img_flat = depth_img.reshape((-1))
        # opencv color to  RGB color between [0, 1)
        colors = real_img[:, :, ::-1]
        colors = colors.reshape((-1, 3)) / 255.
        points_3d = camera.camera2world(point=points, depth=depth_img_flat)
        return points_3d, colors


    def depth2pointcloud_no_color(self, camera, depth_img):
        depth_img = depth_img.transpose()
        points = np.zeros(shape=(depth_img.shape[0], depth_img.shape[1], 2))
        x_range = np.array(range(depth_img.shape[0]))
        y_range = np.array(range(depth_img.shape[1]))
        points[:, :, 0] = np.add(points[:, :, 0].transpose(), x_range).transpose()
        points[:, :, 1] = np.add(points[:, :, 1], y_range)
        points = points.reshape((-1, 2))
        depth_img_flat = depth_img.reshape((-1))
        # opencv color to  RGB color between [0, 1)
        points_3d = camera.camera2world(point=points, depth=depth_img_flat)
        final_point = []
        for point in points_3d:
            if point[2] > 0.1:
                final_point.append(point)
        return final_point

    def __get_img_mask(self, img_width=1280, img_height=1024):
        radius = int(img_height / 2 - 30)
        mask = np.zeros(shape=[img_height, img_width, 3])
        cv2.circle(mask, center=(img_width // 2, img_height // 2), radius=radius, color=(255, 255, 255), thickness=-1)
        return mask / 255.

    def postprocess(self, point_3d, colors):
        final_point = []
        final_color = []
        for point, color in zip(point_3d, colors):
            if point[2] > 0.1:
                final_point.append(point)
                final_color.append(color)
        return final_point, final_color

    def __visualize(self, point_cloud):
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame()
        open3d.visualization.draw_geometries([point_cloud, mesh_frame])

    def get_point_cloud_single_image(self, depth_path, img_path, output_path=None):
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = cv2.resize(depth, dsize=(1280, 1024), interpolation=cv2.INTER_NEAREST)
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        depth[depth > 100] = 0

        img = cv2.imread(img_path)
        img = cv2.resize(img, dsize=(1280, 1024), interpolation=cv2.INTER_LINEAR)

        point_3d, colors = self.depth2pointcloud(self.camera, depth, img)

        if self.post_process:
            point_3d, colors = self.postprocess(point_3d, colors)

        # visualization
        point_cloud = open3d.geometry.PointCloud()
        point_cloud.points = open3d.utility.Vector3dVector(point_3d)
        point_cloud.colors = open3d.utility.Vector3dVector(colors)

        if self.visualization:
            self.__visualize(point_cloud)
        if output_path is not None:
            open3d.io.write_point_cloud(output_path, point_cloud)

        return point_cloud


if __name__ == '__main__':
    root_path = r'\\winfs-inf\CT\EgoMocap\work\EgocentricDepthEstimation\data'
    img_name_map = {'kitchen': 'fc2_save_2017-11-08-124903-0100.jpg',
                    'office': 'fc2_save_2017-11-08-162032-0000.jpg',
                    'img': '004.png',
                    'kitchen_jian': 'img_-04032020183051-58.jpg',
                    'kitchen_2': 'fc2_save_2017-11-08-124903-2300.jpg',
                    'office_2': 'fc2_save_2017-11-08-162032-1300.jpg',
                    'studio': 'frame_c_0_f_0180.png',
                    'office_3': 'fc2_save_2017-11-06-152157-0120.jpg',
                    'me': 'me_processed.jpg',
                    'scannet': '000.png',
                    'kitchen_3': 'fc2_save_2017-11-08-124903-0793.jpg',
                    'real': '0.png',
                    'shakehead': '2.jpg',
                    'kitchen_rot': '1.jpg',
                    'kitchen_seq': 'fc2_save_2017-11-08-124903-0108.jpg',
                    'synthetic_data': '000010.png',
                    'synthetic_data_2': '000000.png',
                    'jian3': 'img-10082020170800-793.jpg'}
    scene_dir = r'kitchen'
    img_name = img_name_map[scene_dir]
    depth_name = img_name + '.exr'
    depth_dir = 'wo_body_lr_1e-4_finetune'
    depth_path = os.path.join(root_path, scene_dir, depth_dir, depth_name)
    img_path = os.path.join(root_path, scene_dir, img_name)

    get_point_cloud = Depth2PointCloud(visualization=True)

    get_point_cloud.get_point_cloud_single_image(depth_path, img_path,
         output_path=os.path.join(r'F:\Develop\egocentricdepthestimation\reconstructed_scene_pointcloud',
                                  '{}_{}_{}.ply'.format(scene_dir, depth_dir, depth_name)))

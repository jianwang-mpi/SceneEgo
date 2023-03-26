import argparse
import sys
sys.path.append('sceneego')
import cv2
from utils.skeleton import Skeleton
import pickle
import open3d

from utils.depth2pointcloud import Depth2PointCloud


def visualize(img_path, depth_path, pred_pose_path):
    skeleton = Skeleton(calibration_path=None)

    with open(pred_pose_path, 'rb') as f:
        predicted_pose = pickle.load(f)

    predicted_pose_mesh = skeleton.joints_2_mesh(predicted_pose)

    get_point_cloud = Depth2PointCloud(visualization=False,
                                       camera_model='utils/fisheye/fisheye.calibration_05_08.json')

    scene = get_point_cloud.get_point_cloud_single_image(depth_path, img_path, output_path=None)

    open3d.visualization.draw_geometries([scene, predicted_pose_mesh])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--pose_path", type=str, required=True)

    args = parser.parse_args()
    img_path = args.img_path
    depth_path = args.depth_path
    pose_path = args.pose_path

    visualize(img_path, depth_path, pose_path)


if __name__ == '__main__':
    main()
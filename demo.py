import os
from pprint import pprint

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.demo_dataset import DemoDataset
from network.voxel_net_depth import VoxelNetwork_depth
from utils import cfg
from utils.skeleton import Skeleton
import argparse
import pickle
from visualize import visualize

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


class Demo:
    def __init__(self, config, img_dir, depth_dir):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.demo_dataset = DemoDataset(config, img_dir, depth_dir, voxel_output=False)
        self.demo_dataloader = DataLoader(self.demo_dataset, batch_size=1, shuffle=False, drop_last=False,
                                          num_workers=0)

        self.network = VoxelNetwork_depth(config)

        # load the network model
        model_path = config.test.model_path
        loads = torch.load(model_path)
        self.network.load_state_dict(loads['state_dict'])

        self.network = self.network.to(self.device)

        self.skeleton = Skeleton(calibration_path=config.dataset.camera_calibration_path)

    def run(self, config):
        print('---------------------Start Training-----------------------')
        pprint(config.__dict__)
        self.network.eval()

        result_list = []
        with torch.no_grad():
            for i, (img, img_rgb, depth_info, img_path) in tqdm(enumerate(self.demo_dataloader)):
                img = img.to(self.device)
                img_rgb = img_rgb.to(self.device)

                depth_info = depth_info.to(self.device)

                grid_coord_proj_batch = self.network.grid_coord_proj_batch
                coord_volumes = self.network.coord_volumes

                vol_keypoints_3d, features, volumes, coord_volumes = self.network(img, grid_coord_proj_batch,
                                                                                  coord_volumes,
                                                                                  depth_map_batch=depth_info)

                predicted_keypoints_batch = vol_keypoints_3d.cpu().numpy()

                assert len(vol_keypoints_3d) == 1 and len(img_path) == 1  # make sure the batch is 1

                result_list.append({'img_path': img_path[0], 'predicted_keypoints': predicted_keypoints_batch[0]})

                # save predicted joint to output dir

        return result_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=False, default='experiments/sceneego/test/sceneego.yaml')
    parser.add_argument("--img_dir", type=str, required=False, default='data/demo/imgs')
    parser.add_argument("--depth_dir", type=str, required=False, default='data/demo/depths')
    parser.add_argument("--output_dir", type=str, required=False, default='data/demo/out')
    parser.add_argument("--vis", type=str, required=False, default='false')

    args = parser.parse_args()

    config_path = args.config
    img_dir = args.img_dir
    depth_dir = args.depth_dir
    output_dir = args.output_dir
    vis = args.vis

    config = cfg.load_config(config_path)
    demo = Demo(config, img_dir, depth_dir)
    result_list = demo.run(config)

    for result_dict in result_list:
        # save predicted joint list
        img_path = result_dict['img_path']
        img_name = os.path.split(img_path)[1]
        pose_pred = result_dict['predicted_keypoints']
        out_path = os.path.join(output_dir, f'{img_name}.pkl')
        depth_path = os.path.join(depth_dir, f'{img_name}.exr')

        with open(out_path, 'wb') as f:
            pickle.dump(pose_pred, f)

        # visualize the pose and depth map
        if vis.lower() == 'true':
            visualize(img_path, depth_path, out_path)


if __name__ == '__main__':
    main()

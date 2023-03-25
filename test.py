from pprint import pprint

import open3d.visualization
import torch
from torch.utils.data import DataLoader

from torch.optim.adam import Adam
from dataset.EgoPWTestDataset_with_real_depth import EgoPWTestDataset_with_depth
from dataset.EgoPWNewTestDataset_with_real_depth import EgoPWNewTestDataset_with_depth
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import datetime
import os
import numpy as np
from utils import cfg
import sys
sys.path.append('..')
from utils_proj.skeleton import Skeleton
from utils_proj.rigid_transform_with_scale import umeyama
from scipy.io import loadmat

from network.voxel_net_depth import VoxelNetwork_depth
from evaluation.python_evaluate_3d_our_dataset import evaluate_3d_our_dataset
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# note: since the round is used when calculating voxelized scene representation, the gradient cannot pass back to
# the depth image
class Test:
    def __init__(self, config, seq_name, estimated_depth_name=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.context_dataset = EgoPWTestDataset_with_depth(config, seq_name=seq_name, local_machine=False,
                                                           voxel_output=False)
        self.context_dataloader = DataLoader(self.context_dataset, batch_size=config.test.batch_size, shuffle=False,
                                             drop_last=False, num_workers=1)

        self.network = VoxelNetwork_depth(config, device=self.device)

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

        predicted_joint_list = []
        gt_joint_list = []
        with torch.no_grad():
            for i, (img, img_rgb, depth_info, img_path) in tqdm(enumerate(self.context_dataloader)):
                img = img.to(self.device)
                img_rgb = img_rgb.to(self.device)

                depth_info = depth_info.to(self.device)

                grid_coord_proj_batch = self.network.grid_coord_proj_batch
                coord_volumes = self.network.coord_volumes

                vol_keypoints_3d, features, volumes, coord_volumes = self.network(img_rgb, grid_coord_proj_batch,
                                                                                  coord_volumes,
                                                                                  depth_map_batch=depth_info)

                predicted_keypoints_batch = vol_keypoints_3d.cpu().numpy()

                predicted_joint_list.extend(predicted_keypoints_batch)
                # print(len(predicted_joint_list))

        return predicted_joint_list



if __name__ == '__main__':
    config_path = 'experiments/egopw_finetune_new_cam_2m/test/egopw_finetune_new_cam_2m.yaml'
    import pickle
    seq_name = 'studio-jian1'

    config = cfg.load_config(config_path)
    test = Test(config, seq_name, estimated_depth_name='finetune_inpaint_network_with_seg_2')
    predicted_joint_list = test.run(config)

    mpjpe, pampjpe = test.context_dataset.evaluate_mpjpe(predicted_joint_list)

    print('mpjpe: {}'.format(mpjpe))
    print('pa mpjpe: {}'.format(pampjpe))


    # save predicted joint list

    # save_dir = r'/HPS/ScanNet/work/egocentric_view/25082022/diogo2/out'
    # if not os.path.isdir(save_dir):
    #     os.makedirs(save_dir)
    # save_path = os.path.join(save_dir, 'estimated_depth_new_diogo2.pkl')
    #
    # save_obj = predicted_joint_list
    # with open(save_path, 'wb') as f:
    #     pickle.dump(save_obj, f)

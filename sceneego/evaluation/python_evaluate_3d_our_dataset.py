from scipy.io import loadmat
import open3d
from utils_proj.skeleton import Skeleton
import numpy as np
from utils_proj.calculate_errors import align_skeleton, calculate_error
import os
from natsort import natsorted
from tqdm import tqdm
import pickle

jian3_motion_type = {
    'start': 557,
    'walking': [np.asarray([557, 897]), np.asarray([1137, 1327])],
    'running': [np.asarray([897, 1137])],
    'boxing': [np.asarray([1327, 1417])],
    'stretching': [np.asarray([1417, 1587])],
    'waving': [np.asarray([1587, 1687])],
    'sitting': [np.asarray([1687, 1857])]}

studio_jian1_motion_type = {
    'start': 503,
    'walking': [np.asarray([503, 1723])],
    'running': [np.asarray([1723, 2153])],
    'crouching': [np.asarray([2153, 2393])],
    'boxing': [np.asarray([2393, 2883])],
    'dancing': [np.asarray([2883, 3223])],
    'stretching': [np.asarray([3223, 3553])],
    'waving': [np.asarray([3553, 3603])]}

studio_lingjie1_motion_type = {
    'start': 551,
    'walking': [np.asarray([551, 1761])],
    'crouching': [np.asarray([1761, 2031])],
    'boxing': [np.asarray([2031, 2331])],
    'dancing': [np.asarray([2331, 2691])],
    'stretching': [np.asarray([2691, 2991])],
    'waving': [np.asarray([2991, 3251])]}

studio_jian2_motion_type = {
    'start': 600,
    'walking': [np.asarray([600, 1920])],
    'dancing': [np.asarray([2170, 2310])],
    'playingballs': [np.asarray([1920, 2170])],
    'opendoor': [np.asarray([2310, 2740])],
    'playgolf': [np.asarray([2740, 2980])],
    'talking': [np.asarray([2980, 3210])],
    'shootingarrow': [np.asarray([3210, 3400])]}

studio_lingjie2_motion_type = {
    'start': 438,
    'walking': [np.asarray([438, 1048])],
    'running': [np.asarray([1048, 1548])],
    'playingballs': [np.asarray([1548, 1748])],
    'opendoor': [np.asarray([1748, 2008])],
    'playgolf': [np.asarray([2008, 2288])],
    'talking': [np.asarray([2288, 2528])],
    'shootingarrow': [np.asarray([2528, 2738])]}

path_dict = {
    'jian3': {
        'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/jian3/jian3.pkl',
        'start_frame': 557,
        'end_frame': 1857,
        "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC08102020/jian3'
    },
    'studio-jian1': {
        'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian1/jian1.pkl',
        'start_frame': 503,
        'end_frame': 3603,
        "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian1'
    },
    'studio-jian2': {
        'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian2/jian2.pkl',
        'start_frame': 600,
        'end_frame': 3400,
        "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian2'
    },
    'studio-lingjie1': {
        'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie1/lingjie1.pkl',
        'start_frame': 551,
        'end_frame': 3251,
        "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie1'
    },
    'studio-lingjie2': {
        'gt_path': r'/HPS/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie2/lingjie2.pkl',
        'start_frame': 438,
        'end_frame': 2738,
        "predicted_path": r'/HPS/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie2'
    }
}


path_dict_local = {
    'jian3': {
        'gt_path': r'X:/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/jian3/jian3.pkl',
        'start_frame': 557,
        'end_frame': 1857,
        "predicted_path": r'X:/Mo2Cap2Plus1/static00/EgocentricData/REC08102020/jian3'
    },
    'studio-jian1': {
        'gt_path': r'X:/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian1/jian1.pkl',
        'start_frame': 503,
        'end_frame': 3603,
        "predicted_path": r'X:/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian1'
    },
    'studio-jian2': {
        'gt_path': r'X:/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-jian2/jian2.pkl',
        'start_frame': 600,
        'end_frame': 3400,
        "predicted_path": r'X:/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-jian2'
    },
    'studio-lingjie1': {
        'gt_path': r'X:/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie1/lingjie1.pkl',
        'start_frame': 551,
        'end_frame': 3251,
        "predicted_path": r'X:/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie1'
    },
    'studio-lingjie2': {
        'gt_path': r'X:/Mo2Cap2Plus/work/MakeWeipengStudioTestData/data/studio-lingjie2/lingjie2.pkl',
        'start_frame': 438,
        'end_frame': 2738,
        "predicted_path": r'X:/Mo2Cap2Plus1/static00/EgocentricData/REC23102020/studio-lingjie2'
    }
}

def evaluate_3d_our_dataset(sequence_name, predicted_pose, scale=True, select_start_to_end=True):
    gt_path = path_dict[sequence_name]['gt_path']
    start_frame = path_dict[sequence_name]['start_frame']
    end_frame = path_dict[sequence_name]['end_frame']
    predicted_path = path_dict[sequence_name]['predicted_path']

    def load_gt_data(gt_path, start_frame, end_frame, mat_start_frame):
        with open(gt_path, 'rb') as f:
            pose_gt = pickle.load(f)
        clip = []
        for i in range(start_frame, end_frame):
            clip.append(pose_gt[i - mat_start_frame])

        skeleton_list = clip

        return np.asarray(skeleton_list)

    gt_pose_list = load_gt_data(gt_path, start_frame, end_frame, start_frame)

    # get predicted pose

    skeleton_model = Skeleton(calibration_path='utils/fisheye/fisheye.calibration.json')

    # sort the predicted poses
    # predicted_pose = natsorted(predicted_pose, key=lambda pose: pose[0])
    #
    # predicted_pose_list = [pose_tuple[1] for pose_tuple in predicted_pose]
    if select_start_to_end:
        predicted_pose_list = predicted_pose[start_frame: end_frame]
    else:
        predicted_pose_list = predicted_pose

    aligned_estimated_result, gt_seq = align_skeleton(predicted_pose_list, gt_pose_list, None, scale=scale)

    aligned_original_mpjpe = calculate_error(aligned_estimated_result, gt_seq)

    # align the estimated result and original result
    aligned_estimated_result, gt_seq = align_skeleton(predicted_pose_list, gt_pose_list, skeleton_model, scale=scale)

    bone_length_aligned_original_mpjpe = calculate_error(aligned_estimated_result, gt_seq)

    print(aligned_original_mpjpe)
    print(bone_length_aligned_original_mpjpe)

    calculate_different_motion(aligned_estimated_result, gt_seq, sequence_name)

    return aligned_original_mpjpe, bone_length_aligned_original_mpjpe


def calculate_different_motion(estimated_pose, gt_pose, data_dir):
    if 'jian3' in data_dir:
        motion_type = jian3_motion_type
    if 'jian1' in data_dir:
        motion_type = studio_jian1_motion_type
    if 'jian2' in data_dir:
        motion_type = studio_jian2_motion_type
    if 'lingjie1' in data_dir:
        motion_type = studio_lingjie1_motion_type
    if 'lingjie2' in data_dir:
        motion_type = studio_lingjie2_motion_type

    skeleton_model = Skeleton(None)
    start_frame = motion_type['start']

    for motion in motion_type.keys():
        if motion == 'start':
            continue
        estimated_mpjpe = 0
        for motion_range in motion_type[motion]:
            estimated_pose_motion = estimated_pose[motion_range[0] - start_frame: motion_range[1] - start_frame]
            gt_pose_motion = gt_pose[motion_range[0] - start_frame: motion_range[1] - start_frame]
            aligned_estimated_result, final_gt_seq = align_skeleton(estimated_pose_motion, gt_pose_motion,
                                                                    skeleton_model)
            estimated_mpjpe += calculate_error(aligned_estimated_result, final_gt_seq)
        estimated_mpjpe /= len(motion_type[motion])
        print("{}: {}".format(motion, estimated_mpjpe))


if __name__ == '__main__':
    evaluate_3d_our_dataset('jian3', heatmap_name='da_external_more_gpu_no_mid_loss',
                            depth_name='finetune_depth_spin_iter_0_depth_5',
                            load_predicted=False)
    evaluate_3d_our_dataset('studio-jian1', heatmap_name='da_external_more_gpu_no_mid_loss',
                            depth_name='finetune_depth_spin_iter_0_depth_5',
                            load_predicted=False)
    evaluate_3d_our_dataset('studio-jian2', heatmap_name='da_external_more_gpu_no_mid_loss',
                            depth_name='finetune_depth_spin_iter_0_depth_5',
                            load_predicted=False)
    evaluate_3d_our_dataset('studio-lingjie1', heatmap_name='da_external_more_gpu_no_mid_loss',
                            depth_name='finetune_depth_spin_iter_0_depth_5',
                            load_predicted=False)
    evaluate_3d_our_dataset('studio-lingjie2', heatmap_name='da_external_more_gpu_no_mid_loss',
                            depth_name='finetune_depth_spin_iter_0_depth_5',
                            load_predicted=False)

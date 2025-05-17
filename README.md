# SceneEgo

Official implementation of paper: 

**Scene-aware Egocentric 3D Human Pose Estimation**

*Jian Wang, Diogo Luvizon, Weipeng Xu, Lingjie Liu, Kripasindhu Sarkar, Christian Theobalt*

*CVPR 2023*

[[Project Page](https://people.mpi-inf.mpg.de/~jianwang/projects/sceneego/)] [[SceneEgo Datasets (Train and Test)](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.VCIHDO)] 

<!-- [[SceneEgo Datasets (Test split)](https://nextcloud.mpi-klsb.mpg.de/index.php/s/q27gwN8tWLMEfrY)] [[SceneEgo Datasets (Train split)](https://nextcloud.mpi-klsb.mpg.de/index.php/s/BsjsMJHBdCxfGt6)] -->

[[EgoGTA](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.MYZMVZ)] [[EgoPW-Scene](https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.EAFCFH)]

![Demo image](./resources/Wang_CVPR_2023.gif)

### Annotation format in Test dataset

The annotation of the dataset is saved in "annotation.pkl" of each sequence. Load the pickle file with:

```python
with open('annotation.pkl', 'rb') as f:
    data = pickle.load(f)
print(data[0].keys())
```
The data is a Python list, each item is a Python dict containing the annotations:
- ext_id: the annotation id of external multiview mocap system; 
- calib_board_pose: the 6d pose of the calibration board on the head;
- ego_pose_gt: the ground truth human body pose under the egocentric camera coordinate system, the joint sequence is: Neck, Right Shoulder, Right Elbow, Right Wrist, Left Shoulder, Left Elbow, Left Wrist, Right Hip, Right Knee, Right Ankle, Right Toe, Left Hip, Left Knee, Left Ankle, Left Toe;
- ext_pose_gt: the human pose ground truth in the mocap system coordinate;
- image_name: name of image under directory "imgs";
- ego_camera_matrix: the 6d pose of the egocentric camera on the head.

The id of the egocentric camera can also be obtained with the synchronization file with:
```python

with open('syn.json', 'r') as f:
    syn_data = json.load(f)

ego_start_frame = syn_data['ego']
ext_start_frame = syn_data['ext']
ego_id = ext_id - ext_start_frame + ego_start_frame
egocentric_image_name = "img_%06d.jpg" % ego_id
```

### Install

1. Create a new anaconda environment

```shell
conda create -n sceneego python=3.9

conda activate sceneego
```

2. Install pytorch 1.13.1 from https://pytorch.org/get-started/previous-versions/

3. Install other dependencies
```shell
pip install -r requirements.txt
```
### Run the demo

1. Download [pre-trained pose estimation model](https://nextcloud.mpi-klsb.mpg.de/index.php/s/DGB6XKEPwwQbmTi) and put it under ```models/sceneego/checkpoints```

2. run:
```shell
python demo.py --config experiments/sceneego/test/sceneego.yaml --img_dir data/demo/imgs --depth_dir data/demo/depths --output_dir data/demo/out --vis True
```
The result will be shown with the open3d visualizer and the predicted pose is saved at ```data/demo/out```.

3. The predicted pose is saved as the pkl file (e.g. ```img_001000.jpg.pkl```). To visualize the predicted result, run:
```shell
python visualize.py --img_path data/demo/imgs/img_001000.jpg --depth_path data/demo/depths/img_001000.jpg.exr --pose_path data/demo/out/img_001000.jpg.pkl
```
The result will be shown with the open3d visualizer.

### Test on your own dataset
If you want to test on your own dataset, after obtaining egocentric frames, you need to:

1. Run the egocentric human body segmentation network to get the human body segmentation for each frame:
   
   See repo: [Egocentric Human Body Segmentation](https://github.com/yt4766269/EgocentricHumanBodySeg)
   
3. Run the depth estimator to get the scene depth map for each frame:

   See repo: [Egocentric Depth Estimator](https://github.com/yt4766269/EgocentricDepthEstimator)


### Citation

If you find this work or code is helpful in your research, please cite:
````
@inproceedings{wang2023scene,
  title={Scene-aware Egocentric 3D Human Pose Estimation},
  author={Wang, Jian and Luvizon, Diogo and Xu, Weipeng and Liu, Lingjie and Sarkar, Kripasindhu and Theobalt, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13031--13040},
  year={2023}
}
````

[//]: # (### Test on real-world dataset)

[//]: # ()
[//]: # (1. Download [pre-trained pose estimation model]&#40;https://nextcloud.mpi-klsb.mpg.de/index.php/s/DGB6XKEPwwQbmTi&#41; and put it under ```models/sceneego/checkpoints```)

[//]: # ()
[//]: # ()
[//]: # (2. Download the test dataset from to ```data/sceneego```)

[//]: # ()
[//]: # (3. run:)

[//]: # (```shell)

[//]: # (python test.py --data_path data/sceneego)

[//]: # (```)







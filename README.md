# SceneEgo

### TODO: This repo is under construction...

Official implementation of paper: 

**Scene-aware Egocentric 3D Human Pose Estimation**

*Jian Wang, Diogo Luvizon, Weipeng Xu, Lingjie Liu, Kripasindhu Sarkar, Christian Theobalt*

*CVPR 2023*

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

1. Download pre-trained pose estimation model from  under ```models```

2. run:
```shell
python demo.py --img data/input/img --depth data/input/depth --out data/output
```
The result will be shown with the open3d visualizer and the predicted pose is saved at ```data/output```.

3. The predicted pose is saved as the pkl file. To visualize the predicted result, run:
```shell
python visualize.py --img data/input/img/1.jpg --depth data/input/depth/1.jpg --out data/output/1.pkl
```
The result will be shown with the open3d visualizer.

### Test on real-world dataset

1. Download pre-trained pose estimation model from  to ```models```

2. Download the test dataset from to ```data/sceneego```

3. run:
```shell
python test.py --data_path data/sceneego
```

4. The pose accuracy will be shown in the shell.






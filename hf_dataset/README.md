---
license: cc-by-nc-4.0
---

This repository contains the synthetic data used in the paper [**iTACO: Interactable Digital Twins of Articulated Objects from Casually Captured RGBD Videos**](https://arxiv.org/abs/2506.08334)

# Term of Use
Our dataset is derived from the PartNet-Mobility dataset. Users are required to agree on the terms of use of the PartNet-Mobility dataset before using our dataset. Researchers shall use our dataset only for non-commercial research and educational purposes.

# File Structure
Inside the sim_data folder, there are several gzip files. origin_data.tar.gz.part* files contain all the generated RGBD videos, as well as ground truth camera poses, joint states, part poses, etc. 
Since the entire dataset comprises 784 videos, it requires significant computation to generate a moving map with [MonST3R](https://github.com/Junyi42/monst3r) and perform [automatic part segmentation](https://github.com/zrporz/AutoSeg-SAM2). 
We also provide the preprocessed data in preprocessing.tar.gz.part* files.

1. After downloading the dataset. You need to merge files before decompressing them.
   ```bash
   cat origin_data.tar.gz.part* > origin_data.tar.gz
   tar zxvf origin_data.tar.gz
   cat preprocessing.tar.gz.part* > preprocessing.tar.gz
   tar zxvf preprocessing.tar.gz
   ```
2. Inside the `partnet_mobility` folder, the file structure looks like this:
   ```
   partnet_mobility
         |__{catgory}
               |__{object id}
                       |__{joint id}
                               |__view_0                 --> generated video data folder
                                     |__depth            --> depth map of each frame
                                     |__rgb              --> rgb images of each frame
                                     |__rgb_reverse      --> rgb images in the reversed order
                                     |__sample_rgb       --> samples of frames we extract for analysis
                                     |__segment          --> actor-level segmentation of each frame, for more details please refer to "https://sapien.ucsd.edu/docs/latest/tutorial/rendering/camera.html#visualize-segmentation"
                                     |__xyz              --> 3D coordinates of each pixel in the image. NOT USED
                                     |__camera_pose.npy  --> ground truth camera poses of each frame in Tx7. The first 3 values are (x,y,z) coordinates, and the remaining 4 values are quaternion in (w, x, y, z) format
                                     |__intrinsics.npy   --> 3x3 intrinsic matrix of the camera
                                     |__video.mp4        --> generated video for visulization purpose
                                     |__video.png        --> object mask of the first frame. NOT USED
                               |__view_1                 --> same structure as view_0
                               |__view_init              --> object surface data
                                     |__rgb              --> object surface from different views
                                     |__segment          --> actor-level segmentation
                                     |__xyz              --> 3D coordinates of each pixel
                               |__actor_pose.pkl         --> 6D poses of each actor of each frame in the video
                               |__gt_joint_value.npy     --> ground truth joint value at each frame
                               |__qpos.npy               --> ground truth joint value of ALL joints of the object at each frame
   
   ```
   One thing to note is the soft links inside `sample_rgb` folder may not be valid. You need to modify them to be valid.
3. Inside the `exp_results/preprocessing` folder, the file structure looks like this:
   ```
   preprocessing
         |__{catgory}
               |__{object id}
                       |__{joint id}
                               |__view_0                             --> preprocessed video data folder
                                     |__monst3r                      --> monst3r results, containing video moving map prediction
                                     |__video_segment_reverse        --> automatic part segmentation of the video
                                                |__small
                                                    |__final-output  --> part segmentations of each frame
                               |__view_1                             --> same structure as view_0
   
   ```
4. In the root directory, `new_partnet_mobility_dataset_correct_intr_meta.json` contains all the ground truth joint parameters such as joint axis, joint positions, joint limits, etc.
   `partnet_mobility_data_split.yaml` contains a list of video paths of different parts of our dataset. `test` part corresponds to the `S-Dataset` in our paper. `train` and `validation` constitute the `L-Dataset` in our paper.
   Although our work does not need any training or fine-tuning, we provide this configuration for possible use by some researchers.

# Citation
If you find our dataset to be helpful, please consider cite our paper
```bibtex
@inproceedings{peng2025itaco,
 booktitle = {3DV 2026},
 author = {Weikun Peng and Jun Lv and Cewu Lu and Manolis Savva},
 title = {{iTACO: Interactable Digital Twins of Articulated Objects from Casually Captured RGBD Videos}},
 year = {2025}
}
```

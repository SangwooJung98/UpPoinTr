## Dataset 

### ColoRadar

We present ColoRadar Maps dataset, with the patches of the radar and the lidar maps generates for different scenes in the ColoRadar dataset. For each scene, the lidar scans are limited to radar FOV and 3D occupancy maps are generated using these scans. The radar maps are generated using the octomap package, but with a radar sensor model. These maps are used to generate patches of smaller regions, to effectively represent them in deep learning approaches. For each radar (input) patch, there exists a lidar(gt) patch, which has points  closest to the radar patch, but denoised, upsampled and complete pointcloud
![dataset](fig/dataset.png)

You can download the dataset from: ![Onedrive](https://o365coloradoedu-my.sharepoint.com/:u:/g/personal/ajmo2266_colorado_edu/EZAvY-Bh1-hEn7I9gtUjno4B-WnbQPEyG2RMkOcX59Sjmg?e=v3Nroh)



### Data Preparation
The overall directory structure should be:

```
│PoinTr/
├──cfgs/
├──datasets/
├──data/
│   ├──ShapeNet55-34/
│   ├──PCN/
│   ├──KITTI/
|   ├──ColoRadar/
|      ├──gt/
|         ├──{envName}_{sceneId}_{patchID}.pcd
|      ├──input/
|         ├──{envName}_{sceneId}_{patchID}.pcd
|      ├──train.txt
|      ├──test.txt
├──.......
```
envName - arpg_lab/ aspen/ ec_hallways/ edgar_army/ edgar_classroom/ outdoors
sceneID - run{id}, run0 from each environment is considered as test data




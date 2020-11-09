import pcl
import os

lst = os.listdir('/home/imlb/Desktop/testing/velodyne')
for i,fil in enumerate(lst):
    fl = pcl.load('/home/imlb/Desktop/testing/velodyne/' + fil)
    pcl.save(fl,'/home/imlb/Desktop/lidar_kitti_ply/' + str(i) + '.ply')


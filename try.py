import numpy as np 


# a = '/mnt/retina/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/id05654/6UV8K4t_6Xw/00021.mp4'
a = '/home/cxu-serve/p1/lchen63/voxceleb/unzip/test_video/id05654/6UV8K4t_6Xw/00021.npy'

b = np.load(a)
print (b.shape)
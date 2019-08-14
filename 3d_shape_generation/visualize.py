import os
import numpy as np
import scipy.io as io
import scipy.ndimage as nd
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_dir = '../Datasets/3DShapeNets/volumetric_data/'
get_object_name = lambda : random.choice( os.listdir(data_dir) )

def get_file_path(object_name):
	dir_path = data_dir + str(object_name) + '/30/train'
	file_name = random.choice( os.listdir(dir_path) )
	return os.path.join(dir_path, file_name)

def get_voxels_from_mat(path):
	voxels = io.loadmat(path)['instance']
	voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
	voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
	return voxels

def plot_voxels(voxels, filepath=None, voxels_class=None, show=False):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	ax.set_aspect('equal')
	ax.voxels(voxels, edgecolor='red')
	if voxels_class:
		ax.set_title(voxels_class)
	if show:
		plt.show()
	if filepath:
		plt.savefig(filepath)
	plt.close(fig)

def get_3images_for_category(obj='airplane', train=True, cube_len=64, obj_ratio=1.0):
	obj_path = data_dir + obj + '/30/'
	obj_path += 'train/' if train else 'test/'
	filelist = [f for f in os.listdir(obj_path) if f.endswith('.mat')]
	filelist = filelist[0: int(obj_ratio * len(filelist))]
	volume_batch = np.asarray( [get_voxels_from_mat(obj_path + f) for f in filelist], dtype=np.bool )
	return volume_batch



if __name__ == '__main__':
	
	object_name = get_object_name()
	file_path = get_file_path(object_name)
	voxels = get_voxels_from_mat(file_path)
	plot_voxels(voxels, voxels_class=object_name, filepath=os.path.join('results', str(object_name)+'.png'))

	volumes = get_3images_for_category(obj='airplane', train=True, obj_ratio=1.0)
	volumes = volumes[..., np.newaxis].astype(np.float)
	print(volumes.shape)
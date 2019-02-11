import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class ScanNet(data.Dataset):
	"""ScanNet dataset http://www.scan-net.org/

	Keyword arguments:
	- root_dir (``string``): Path to the base directory of the dataset
	- scene_file (``string``): Path to file containing a list of scenes to be loaded
	- transform (``callable``, optional): A function/transform that takes in a 
	PIL image and returns a transformed version of the image. Default: None.
	- label_transform (``callable``, optional): A function/transform that takes 
	in the target and transforms it. Default: None.
	- loader (``callable``, optional): A function to load an image given its path.
	By default, ``default_loader`` is used.
	- color_mean (``list``): A list of length 3, containing the R, G, B channelwise mean.
	- color_std (``list``): A list of length 3, containing the R, G, B channelwise standard deviation.
	- load_depth (``bool``): Whether or not to load depth images (architectures that use depth 
	information need depth to be loaded).
	- seg_classes (``string``): The palette of classes that the network should learn.
	"""

	def __init__(self, root_dir, scene_file, mode='train', transform=None, label_transform = None, \
		loader=utils.scannet_loader, color_mean=[0.,0.,0.], color_std=[1.,1.,1.], load_depth=False, \
		seg_classes='nyu40'):
		
		self.root_dir = root_dir
		self.scene_file = scene_file
		self.mode = mode
		self.transform = transform
		self.label_transform = label_transform
		self.loader = loader
		self.length = 0
		self.color_mean = color_mean
		self.color_std = color_std
		self.load_depth = load_depth
		self.seg_classes = seg_classes
		# color_encoding has to be initialized AFTER seg_classes
		self.color_encoding = self.get_color_encoding()

		if self.load_depth is True:
			self.loader = utils.scannet_loader_depth

		# Get the list of scenes, and generate paths
		scene_list = []
		try:
			scene_file = open(self.scene_file, 'r')
			scenes = scene_file.readlines()
			scene_file.close()
			for scene in scenes:
				scene = scene.strip().split()
				scene_list.append(scene[0])
		except Exception as e:
			raise e

		if self.mode.lower() == 'train':
			# Get train data and labels filepaths
			self.train_data = []
			self.train_depth = []
			self.train_labels = []
			for scene in scene_list:
				color_images, depth_images, labels = utils.get_filenames_scannet(self.root_dir, scene)
				self.train_data += color_images
				self.train_depth += depth_images
				self.train_labels += labels
				self.length += len(color_images)
		elif self.mode.lower() == 'val':
			# Get val data and labels filepaths
			self.val_data = []
			self.val_depth = []
			self.val_labels = []
			for scene in scene_list:
				color_images, depth_images, labels = utils.get_filenames_scannet(self.root_dir, scene)
				self.val_data += color_images
				self.val_depth += depth_images
				self.val_labels += labels
				self.length += len(color_images)
		elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
			# Get test data and labels filepaths
			self.test_data = []
			self.test_depth = []
			self.test_labels = []
			for scene in scene_list:
				color_images, depth_images, labels = utils.get_filenames_scannet(self.root_dir, scene)
				self.test_data += color_images
				self.test_depth += depth_images
				self.test_labels += labels
				self.length += len(color_images)
		else:
			raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test, inference')


	def __getitem__(self, index):
		""" Returns element at index in the dataset.

		Args:
		- index (``int``): index of the item in the dataset

		Returns:
		A tuple of ``PIL.Image`` (image, label) where label is the ground-truth of the image

		"""

		if self.load_depth is True:

			if self.mode.lower() == 'train':
				data_path, depth_path, label_path = self.train_data[index], self.train_depth[index], \
													self.train_labels[index]
			elif self.mode.lower() == 'val':
				data_path, depth_path, label_path = self.val_data[index], self.val_depth[index], \
													self.val_labels[index]
			elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
				data_path, depth_path, label_path = self.test_data[index], self.test_depth[index], \
													self.test_labels[index]
			else:
				raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test, inference')

			rgbd, label = self.loader(data_path, depth_path, label_path, self.color_mean, self.color_std, \
				self.seg_classes)

			if self.mode.lower() == 'inference':
				return rgbd, label, data_path, depth_path, label_path
			else:
				return rgbd, label

		else:

			if self.mode.lower() == 'train':
				data_path, label_path = self.train_data[index], self.train_labels[index]
			elif self.mode.lower() == 'val':
				data_path, label_path = self.val_data[index], self.val_labels[index]
			elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
				data_path, label_path = self.test_data[index], self.test_labels[index]
			else:
				raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')

			img, label = self.loader(data_path, label_path, self.color_mean, self.color_std, self.seg_classes)

			if self.mode.lower() == 'inference':
				return img, label, data_path, label_path
			else:
				return img, label


	def __len__(self):
		""" Returns the length of the dataset. """
		return self.length


	def get_color_encoding(self):
		if self.seg_classes.lower() == 'nyu40':
			"""Color palette for nyu40 labels """
			return OrderedDict([
				('unlabeled', (0, 0, 0)),
				('wall', (174, 199, 232)),
				('floor', (152, 223, 138)),
				('cabinet', (31, 119, 180)),
				('bed', (255, 187, 120)),
				('chair', (188, 189, 34)),
				('sofa', (140, 86, 75)),
				('table', (255, 152, 150)),
				('door', (214, 39, 40)),
				('window', (197, 176, 213)),
				('bookshelf', (148, 103, 189)),
				('picture', (196, 156, 148)),
				('counter', (23, 190, 207)),
				('blinds', (178, 76, 76)),
				('desk', (247, 182, 210)),
				('shelves', (66, 188, 102)),
				('curtain', (219, 219, 141)),
				('dresser', (140, 57, 197)),
				('pillow', (202, 185, 52)),
				('mirror', (51, 176, 203)),
				('floormat', (200, 54, 131)),
				('clothes', (92, 193, 61)),
				('ceiling', (78, 71, 183)),
				('books', (172, 114, 82)),
				('refrigerator', (255, 127, 14)),
				('television', (91, 163, 138)),
				('paper', (153, 98, 156)),
				('towel', (140, 153, 101)),
				('showercurtain', (158, 218, 229)),
				('box', (100, 125, 154)),
				('whiteboard', (178, 127, 135)),
				('person', (120, 185, 128)),
				('nightstand', (146, 111, 194)),
				('toilet', (44, 160, 44)),
				('sink', (112, 128, 144)),
				('lamp', (96, 207, 209)),
				('bathtub', (227, 119, 194)),
				('bag', (213, 92, 176)),
				('otherstructure', (94, 106, 211)),
				('otherfurniture', (82, 84, 163)),
				('otherprop', (100, 85, 144)),
			])
		elif self.seg_classes.lower() == 'scannet20':
			return OrderedDict([
				('unlabeled', (0, 0, 0)),
				('wall', (174, 199, 232)),
				('floor', (152, 223, 138)),
				('cabinet', (31, 119, 180)),
				('bed', (255, 187, 120)),
				('chair', (188, 189, 34)),
				('sofa', (140, 86, 75)),
				('table', (255, 152, 150)),
				('door', (214, 39, 40)),
				('window', (197, 176, 213)),
				('bookshelf', (148, 103, 189)),
				('picture', (196, 156, 148)),
				('counter', (23, 190, 207)),
				('desk', (247, 182, 210)),
				('curtain', (219, 219, 141)),
				('refrigerator', (255, 127, 14)),
				('showercurtain', (158, 218, 229)),
				('toilet', (44, 160, 44)),
				('sink', (112, 128, 144)),
				('bathtub', (227, 119, 194)),
				('otherfurniture', (82, 84, 163)),
			])

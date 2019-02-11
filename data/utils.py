import os
from PIL import Image
from natsort import natsorted
import numpy as np
import imageio

import torch
import torchvision.transforms as transforms


def get_filenames_scannet(base_dir, scene_id):
	"""Helper function that returns a list of scannet images and the corresponding 
	segmentation labels, given a base directory name and a scene id.

	Args:
	- base_dir (``string``): Path to the base directory containing ScanNet data, in the 
	directory structure specified in https://github.com/angeladai/3DMV/tree/master/prepare_data
	- scene_id (``string``): ScanNet scene id

	"""

	if not os.path.isdir(base_dir):
		raise RuntimeError('\'{0}\' is not a directory.'.format(base_dir))

	color_images = []
	depth_images = []
	labels = []

	# Explore the directory tree to get a list of all files
	for path, _, files in os.walk(os.path.join(base_dir, scene_id, 'color')):
		files = natsorted(files)
		for file in files:
			filename, _ = os.path.splitext(file)
			depthfile = os.path.join(base_dir, scene_id, 'depth', filename + '.png')
			labelfile = os.path.join(base_dir, scene_id, 'label', filename + '.png')
			# Add this file to the list of train samples, only if its corresponding depth and label 
			# files exist.
			if os.path.exists(depthfile) and os.path.exists(labelfile):
				color_images.append(os.path.join(base_dir, scene_id, 'color', filename + '.jpg'))
				depth_images.append(depthfile)
				labels.append(labelfile)

	# Assert that we have the same number of color, depth images as labels
	assert (len(color_images) == len(depth_images) == len(labels))

	return color_images, depth_images, labels


def get_files(folder, name_filter=None, extension_filter=None):
	"""Helper function that returns the list of files in a specified folder
	with a specified extension.

	Keyword arguments:
	- folder (``string``): The path to a folder.
	- name_filter (```string``, optional): The returned files must contain
	this substring in their filename. Default: None; files are not filtered.
	- extension_filter (``string``, optional): The desired file extension.
	Default: None; files are not filtered

	"""
	if not os.path.isdir(folder):
		raise RuntimeError("\"{0}\" is not a folder.".format(folder))

	# Filename filter: if not specified don't filter (condition always true);
	# otherwise, use a lambda expression to filter out files that do not
	# contain "name_filter"
	if name_filter is None:
		# This looks hackish...there is probably a better way
		name_cond = lambda filename: True
	else:
		name_cond = lambda filename: name_filter in filename

	# Extension filter: if not specified don't filter (condition always true);
	# otherwise, use a lambda expression to filter out files whose extension
	# is not "extension_filter"
	if extension_filter is None:
		# This looks hackish...there is probably a better way
		ext_cond = lambda filename: True
	else:
		ext_cond = lambda filename: filename.endswith(extension_filter)

	filtered_files = []

	# Explore the directory tree to get files that contain "name_filter" and
	# with extension "extension_filter"
	for path, _, files in os.walk(folder):
		files.sort()
		for file in files:
			if name_cond(file) and ext_cond(file):
				full_path = os.path.join(path, file)
				filtered_files.append(full_path)

	return filtered_files


def scannet_loader(data_path, label_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.], seg_classes='nyu40'):
	"""Loads a sample and label image given their path as PIL images. (nyu40 classes)

	Keyword arguments:
	- data_path (``string``): The filepath to the image.
	- label_path (``string``): The filepath to the ground-truth image.
	- color_mean (``list``): R, G, B channel-wise mean
	- color_std (``list``): R, G, B channel-wise stddev
	- seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

	Returns the image and the label as PIL images.

	"""

	# Load image
	data = np.array(imageio.imread(data_path))
	# Reshape data from H x W x C to C x H x W
	data = np.moveaxis(data, 2, 0)
	# Define normalizing transform
	normalize = transforms.Normalize(mean=color_mean, std=color_std)
	# Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
	data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))

	# Load label
	if seg_classes.lower() == 'nyu40':
		label = np.array(imageio.imread(label_path)).astype(np.uint8)
	elif seg_classes.lower() == 'scannet20':
		label = np.array(imageio.imread(label_path)).astype(np.uint8)
		# Remap classes from 'nyu40' to 'scannet20'
		label = nyu40_to_scannet20(label)

	return data, label


def scannet_loader_depth(data_path, depth_path, label_path, color_mean=[0.,0.,0.], color_std=[1.,1.,1.], \
	seg_classes='nyu40'):
	"""Loads a sample and label image given their path as PIL images. (nyu40 classes)

	Keyword arguments:
	- data_path (``string``): The filepath to the image.
	- depth_path (``string``): The filepath to the depth png.
	- label_path (``string``): The filepath to the ground-truth image.
	- color_mean (``list``): R, G, B channel-wise mean
	- color_std (``list``): R, G, B channel-wise stddev
	- seg_classes (``string``): Palette of classes to load labels for ('nyu40' or 'scannet20')

	Returns the image and the label as PIL images.

	"""

	# Load image
	rgb = np.array(imageio.imread(data_path))
	# Reshape rgb from H x W x C to C x H x W
	rgb = np.moveaxis(rgb, 2, 0)
	# Define normalizing transform
	normalize = transforms.Normalize(mean=color_mean, std=color_std)
	# Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
	rgb = normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))

	# Load depth
	depth = torch.Tensor(np.array(imageio.imread(depth_path)).astype(np.float32) / 1000.0)
	depth = torch.unsqueeze(depth, 0)

	# Concatenate rgb and depth
	data = torch.cat((rgb, depth), 0)

	# Load label
	if seg_classes.lower() == 'nyu40':
		label = np.array(imageio.imread(label_path)).astype(np.uint8)
	elif seg_classes.lower() == 'scannet20':
		label = np.array(imageio.imread(label_path)).astype(np.uint8)
		# Remap classes from 'nyu40' to 'scannet20'
		label = nyu40_to_scannet20(label)

	return data, label


def nyu40_to_scannet20(label):
	"""Remap a label image from the 'nyu40' class palette to the 'scannet20' class palette """

	# Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
	# Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in 
	# 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
	# http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
	# http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
	# The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

	# The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the 
	# corresponding target 'scannet20' label
	remapping = [(0,0),(13,0),(15,0),(17,0),(18,0),(19,0),(20,0),(21,0),(22,0),(23,0),(25,0),(26,0),(27,0),
				(29,0),(30,0),(31,0),(32,0),(35,0),(37,0),(38,0),(40,0),(14,13),(16,14),(24,15),(28,16),(33,17),
				(34,18),(36,19),(39,20)]
	for src, tar in remapping:
		label[np.where(label==src)] = tar
	return label


def create_label_image(output, color_palette):
	"""Create a label image, given a network output (each pixel contains class index) and a color palette.

	Args:
	- output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
	corresponding to the class label of that pixel.
	- color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
	"""
	
	label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
	for idx, color in enumerate(color_palette):
		label_image[output==idx] = color
	return label_image


def remap(image, old_values, new_values):
	assert isinstance(image, Image.Image) or isinstance(
		image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
	assert type(new_values) is tuple, "new_values must be of type tuple"
	assert type(old_values) is tuple, "old_values must be of type tuple"
	assert len(new_values) == len(
		old_values), "new_values and old_values must have the same length"

	# If image is a PIL.Image convert it to a numpy array
	if isinstance(image, Image.Image):
		image = np.array(image)

	# Replace old values by the new ones
	tmp = np.zeros_like(image)
	for old, new in zip(old_values, new_values):
		# Since tmp is already initialized as zeros we can skip new values
		# equal to 0
		if new != 0:
			tmp[image == old] = new

	return Image.fromarray(tmp)


def enet_weighing(dataloader, num_classes, c=1.02):
	"""Computes class weights as described in the ENet paper:

		w_class = 1 / (ln(c + p_class)),

	where c is usually 1.02 and p_class is the propensity score of that
	class:

		propensity_score = freq_class / total_pixels.

	References: https://arxiv.org/abs/1606.02147

	Keyword arguments:
	- dataloader (``data.Dataloader``): A data loader to iterate over the
	dataset.
	- num_classes (``int``): The number of classes.
	- c (``int``, optional): AN additional hyper-parameter which restricts
	the interval of values for the weights. Default: 1.02.

	"""
	class_count = 0
	total = 0
	for _, label in dataloader:
		label = label.cpu().numpy()

		# Flatten label
		flat_label = label.flatten()

		# Sum up the number of pixels of each class and the total pixel
		# counts for each label
		class_count += np.bincount(flat_label, minlength=num_classes)
		total += flat_label.size

	# Compute propensity score and then the weights for each class
	propensity_score = class_count / total
	class_weights = 1 / (np.log(c + propensity_score))

	return class_weights


def median_freq_balancing(dataloader, num_classes):
	"""Computes class weights using median frequency balancing as described
	in https://arxiv.org/abs/1411.4734:

		w_class = median_freq / freq_class,

	where freq_class is the number of pixels of a given class divided by
	the total number of pixels in images where that class is present, and
	median_freq is the median of freq_class.

	Keyword arguments:
	- dataloader (``data.Dataloader``): A data loader to iterate over the
	dataset.
	whose weights are going to be computed.
	- num_classes (``int``): The number of classes

	"""
	class_count = 0
	total = 0
	for _, label in dataloader:
		label = label.cpu().numpy()

		# Flatten label
		flat_label = label.flatten()

		# Sum up the class frequencies
		bincount = np.bincount(flat_label, minlength=num_classes)

		# Create of mask of classes that exist in the label
		mask = bincount > 0
		# Multiply the mask by the pixel count. The resulting array has
		# one element for each class. The value is either 0 (if the class
		# does not exist in the label) or equal to the pixel count (if
		# the class exists in the label)
		total += mask * flat_label.size

		# Sum up the number of pixels found for each class
		class_count += bincount

	# Compute the frequency and its median
	freq = class_count / total
	med = np.median(freq)

	return med / freq

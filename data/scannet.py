import os
from collections import OrderedDict
import torch.utils.data as data
from . import utils


class ScanNet(data.Dataset):
	"""ScanNet dataset http://www.scan-net.org/

	Keyword arguments:
	- root_dir (``string``): Path to the base directory of the dataset
	- mode (``string``): 'train', 'val', or 'test'
	- transform (``callable``, optional): A function/transform that takes in a 
	PIL image and returns a transformed version of the image. Default: None.
	- label_transform (``callable``, optional): A function/transform that takes 
	in the target and transforms it. Default: None.
	- loader (``callable``, optional): A function to load an image given its path.
	By default, ``default_loader`` is used.
	"""

	def __init__(self, root_dir, mode='train', transform=None, label_transform = None, loader=utils.pil_loader):
		
		self.root_dir = root_dir
		self.mode = mode
		self.transform = transform
		self.label_transform = label_transform
		self.loader = loader


	def __getitem__(self, index):
		pass


	def __len__(self, index):
		return 0

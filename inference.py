import os
import time

import imageio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data
import torchvision.transforms as transforms

import transforms as ext_transforms
from models.enet import ENet, ENetDepth
from metric.iou import IoU
from args import get_arguments
from data.utils import enet_weighing, median_freq_balancing
import utils


# Get the arguments
args = get_arguments()

device = torch.device(args.device)

# Mean color, standard deviation (R, G, B)
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]


class Inference:
	"""Runs Inference using the ``model`` on the specified test dataset using the
	data loader, and loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to run inference on.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.

	"""

	def __init__(self, model, data_loader, criterion, metric, device, arch='rgb', generate_images=False, \
		color_palette=None):

		self.model = model
		self.data_loader = data_loader
		self.criterion = criterion
		self.metric = metric
		self.device = device
		self.arch = arch
		self.generate_images = generate_images
		if self.generate_images is True:
			if not os.path.exists(os.path.join(args.save_dir, args.name + '_images')):
				os.makedirs(os.path.join(args.save_dir, args.name + '_images'))
				print('Created directory:', os.path.join(args.save_dir, args.name + '_images'))
			self.generate_image_dir = os.path.join(args.save_dir, args.name + '_images')
			self.color_palette = color_palette


	def run_epoch(self, iteration_loss=False):
		"""Runs an epoch of validation.

		Keyword arguments:
		- iteration_loss (``bool``, optional): Prints loss at every step.

		Returns:
		- The epoch loss (float), and the values of the specified metrics

		"""
		self.model.eval()
		epoch_loss = 0.0
		self.metric.reset()
		avgTime = 0.0
		numTimeSteps = 0
		fileName = 0
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			# Get the inputs and labels
			inputs = batch_data[0].to(self.device)
			labels = batch_data[1].long().to(self.device)

			# Operate differently for 'rgb' vs 'rgbd' architectures
			if self.arch.lower() == 'rgb':
				data_path = batch_data[2]
				label_path = batch_data[3]
			elif self.arch.lower() == 'rgbd':
				data_path = batch_data[2]
				depth_path = batch_data[3]
				label_path = batch_data[4]
			else:
				raise RuntimeError('Invalid architecture specified.')

			with torch.no_grad():
				# Forward propagation
				outputs = self.model(inputs)

				# Loss computation
				loss = self.criterion(outputs, labels)

				if self.generate_images is True:
					bs = len(data_path)
					# Collapse the outputs by taking max along class dimension 
					for b in range(bs):
						cur_rgb = data_path[b]
						cur_label = label_path[b]
						cur_output = torch.clone(outputs[b])
						_, cur_output = cur_output.max(0)
						cur_output = cur_output.detach().cpu().numpy()
						pred_label_image = create_label_image(cur_output, self.color_palette)
						gt_label_image = torch.clone(labels[b]).detach().cpu().numpy()
						gt_label_image = create_label_image(gt_label_image, self.color_palette)
						rgb_image = imageio.imread(cur_rgb)

						height = cur_output.shape[0]
						width = cur_output.shape[1]
						composite_image = np.zeros((3*height, width, 3), dtype=np.uint8)
						composite_image[0:height,:,:] = rgb_image
						composite_image[height:2*height,:,:] = pred_label_image
						composite_image[2*height:,:,:] = gt_label_image
						imageio.imwrite(os.path.join(self.generate_image_dir, str(fileName)+'.png'), \
							composite_image)
						fileName += 1

						# imageio.imwrite(os.path.join(self.generate_image_dir, 'rgb.png'), rgb_image)
						# imageio.imwrite(os.path.join(self.generate_image_dir, 'pred.png'), pred_label_image)
						# imageio.imwrite(os.path.join(self.generate_image_dir, 'gt.png'), gt_label_image)

			# Keep track of loss for current epoch
			epoch_loss += loss.item()

			# Keep track of evaluation the metric
			self.metric.add(outputs.detach(), labels.detach())
			endTime = time.time()
			avgTime += (endTime - startTime)
			numTimeSteps += 1

			if iteration_loss > 0 and (step % iteration_loss == 0):
				print("[Step: %d/%d (%3.2f ms)] Iteration loss: %.4f" % (step, len(self.data_loader), \
					1000*(avgTime / (numTimeSteps if numTimeSteps>0 else 1)), loss.item()))
				numTimeSteps = 0
				avgTime = 0.

		return epoch_loss / len(self.data_loader), self.metric.value()


def create_label_image(output, color_palette):
	"""Create a label image, given a network output (each pixel contains class index) and a color palette.

	Args:
	- output (``np.array``, dtype = np.uint8): Output image. Height x Width. Each pixel contains an integer, 
	corresponding to the class label of that pixel.
	- color_palette (``OrderedDict``): Contains (R, G, B) colors (uint8) for each class.
	"""
	
	label_image = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
	for idx, color in enumerate(color_palette):
		label_image[output==idx] = color_palette[color]
	return label_image


def load_dataset(dataset):
	print("\nLoading dataset...\n")

	print("Selected dataset:", args.dataset)
	print("Dataset directory:", args.dataset_dir)
	print('Test file:', args.testFile)
	print("Save directory:", args.save_dir)

	image_transform = transforms.Compose(
		[transforms.Resize((args.height, args.width)),
		 transforms.ToTensor()])

	label_transform = transforms.Compose([
		transforms.Resize((args.height, args.width)),
		ext_transforms.PILToLongTensor()
	])

	# Load the test set as tensors
	test_set = dataset(args.dataset_dir, args.testFile, mode='inference', transform=image_transform, \
		label_transform=label_transform, color_mean=color_mean, color_std=color_std, \
		load_depth=(args.arch=='rgbd'), seg_classes=args.seg_classes)
	test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

	# Get encoding between pixel valus in label images and RGB colors
	class_encoding = test_set.color_encoding

	# Get number of classes to predict
	num_classes = len(class_encoding)

	# Print information for debugging
	print("Number of classes to predict:", num_classes)
	print("Test dataset size:", len(test_set))

	# Get a batch of samples to display
	if args.arch == 'rgbd':
		images, labels, data_path, depth_path, label_path = iter(test_loader).next()
	else:
		images, labels, data_path, label_path = iter(test_loader).next()

	print("Image size:", images.size())
	print("Label size:", labels.size())
	print("Class-color encoding:", class_encoding)

	# Show a batch of samples and labels
	if args.imshow_batch:
		print("Close the figure window to continue...")
		label_to_rgb = transforms.Compose([
			ext_transforms.LongTensorToRGBPIL(class_encoding),
			transforms.ToTensor()
		])
		color_labels = utils.batch_transform(labels, label_to_rgb)
		utils.imshow_batch(images, color_labels)

	# Get class weights
	# If a class weight file is provided, try loading weights from in there
	class_weights = None
	if args.class_weights_file:
		print('Trying to load class weights from file...')
		try:
			class_weights = np.loadtxt(args.class_weights_file)
		except Exception as e:
			raise e
	else:
		print('No class weights found...')

	if class_weights is not None:
		class_weights = torch.from_numpy(class_weights).float().to(device)
		# Set the weight of the unlabeled class to 0
		if args.ignore_unlabeled:
			ignore_index = list(class_encoding).index('unlabeled')
			class_weights[ignore_index] = 0

	print("Class weights:", class_weights)

	return test_loader, class_weights, class_encoding


def inference(model, test_loader, class_weights, class_encoding):
	print("\nInference...\n")

	num_classes = len(class_encoding)

	# We are going to use the CrossEntropyLoss loss function as it's most
	# frequentely used in classification problems with multiple classes which
	# fits the problem. This criterion  combines LogSoftMax and NLLLoss.
	criterion = nn.CrossEntropyLoss(weight=class_weights)

	# Evaluation metric
	if args.ignore_unlabeled:
		ignore_index = list(class_encoding).index('unlabeled')
	else:
		ignore_index = None
	metric = IoU(num_classes, ignore_index=ignore_index)

	# Test the trained model on the test set
	test = Inference(model, test_loader, criterion, metric, device, args.arch, \
		generate_images=args.generate_images, color_palette=class_encoding)

	print(">>>> Running test dataset")

	loss, (iou, miou) = test.run_epoch(args.print_step)
	class_iou = dict(zip(class_encoding.keys(), iou))

	print(">>>> Avg. loss: {0:.4f} | Mean IoU: {1:.4f}".format(loss, miou))

	# Print per class IoU
	for key, class_iou in zip(class_encoding.keys(), iou):
		print("{0}: {1:.4f}".format(key, class_iou))

	# Show a batch of samples and labels
	if args.imshow_batch:
		print("A batch of predictions from the test set...")
		images, _ = iter(test_loader).next()
		predict(model, images, class_encoding)


def predict(model, images, class_encoding):
	images = images.to(device)

	# Make predictions!
	model.eval()
	with torch.no_grad():
		predictions = model(images)

	# Predictions is one-hot encoded with "num_classes" channels.
	# Convert it to a single int using the indices where the maximum (1) occurs
	_, predictions = torch.max(predictions.data, 1)

	label_to_rgb = transforms.Compose([
		ext_transforms.LongTensorToRGBPIL(class_encoding),
		transforms.ToTensor()
	])
	color_predictions = utils.batch_transform(predictions.cpu(), label_to_rgb)
	utils.imshow_batch(images.data.cpu(), color_predictions)


# Run only if this module is being run directly
if __name__ == '__main__':

	# Fail fast if the dataset directory doesn't exist
	assert os.path.isdir(
		args.dataset_dir), "The directory \"{0}\" doesn't exist.".format(
			args.dataset_dir)

	# Fail fast if the saving directory doesn't exist
	assert os.path.isdir(
		args.save_dir), "The directory \"{0}\" doesn't exist.".format(
			args.save_dir)

	# Import the requested dataset
	if args.dataset.lower() == 'scannet':
		from data import ScanNet as dataset
	else:
		# Should never happen...but just in case it does
		raise RuntimeError("\"{0}\" is not a supported dataset.".format(
			args.dataset))

	test_loader, w_class, class_encoding = load_dataset(dataset)

	# Intialize a new ENet model
	num_classes = len(class_encoding)
	if args.arch.lower() == 'rgb':
		model = ENet(num_classes).to(device)
	elif args.arch.lower() == 'rgbd':
		model = ENetDepth(num_classes).to(device)
	else:
		# This condition will not occur (argparse will fail if an invalid option is specified)
		raise RuntimeError('Invalid network architecture specified.')

	# Initialize a optimizer just so we can retrieve the model from the
	# checkpoint
	optimizer = optim.Adam(model.parameters())

	# Load the previoulsy saved model state to the ENet model
	model = utils.load_checkpoint(model, optimizer, args.save_dir,
								  args.name)[0]
	# print(model)
	inference(model, test_loader, w_class, class_encoding)
	
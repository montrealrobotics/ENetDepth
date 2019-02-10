import torch
import time


class Test:
	"""Tests the ``model`` on the specified test dataset using the
	data loader, and loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to test.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.

	"""

	def __init__(self, model, data_loader, criterion, metric, device):
		self.model = model
		self.data_loader = data_loader
		self.criterion = criterion
		self.metric = metric
		self.device = device

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
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			# Get the inputs and labels
			inputs = batch_data[0].to(self.device)
			labels = batch_data[1].long().to(self.device)

			with torch.no_grad():
				# Forward propagation
				outputs = self.model(inputs)

				# Loss computation
				loss = self.criterion(outputs, labels)

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

import time

class Train:
	"""Performs the training of ``model`` given a training dataset data
	loader, the optimizer, and the loss criterion.

	Keyword arguments:
	- model (``nn.Module``): the model instance to train.
	- data_loader (``Dataloader``): Provides single or multi-process
	iterators over the dataset.
	- optim (``Optimizer``): The optimization algorithm.
	- criterion (``Optimizer``): The loss criterion.
	- metric (```Metric``): An instance specifying the metric to return.
	- device (``torch.device``): An object representing the device on which
	tensors are allocated.

	"""

	def __init__(self, model, data_loader, optim, criterion, metric, device):
		self.model = model
		self.data_loader = data_loader
		self.optim = optim
		self.criterion = criterion
		self.metric = metric
		self.device = device

	def run_epoch(self, iteration_loss=0):
		"""Runs an epoch of training.

		Keyword arguments:
		- iteration_loss (``bool``, optional): Prints loss at every step.

		Returns:
		- The epoch loss (float).

		"""
		self.model.train()
		epoch_loss = 0.0
		self.metric.reset()
		avgTime = 0.0
		numTimeSteps = 0
		for step, batch_data in enumerate(self.data_loader):
			startTime = time.time()
			# Get the inputs and labels
			inputs = batch_data[0].to(self.device)
			labels = batch_data[1].long().to(self.device)

			# Forward propagation
			outputs = self.model(inputs)

			# Loss computation
			loss = self.criterion(outputs, labels)

			# Backpropagation
			self.optim.zero_grad()
			loss.backward()
			self.optim.step()

			# Keep track of loss for current epoch
			epoch_loss += loss.item()

			# Keep track of the evaluation metric
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class FcNet(nn.Module):
	"""
	Fully connected network for MNIST classification
	"""

	def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

		super().__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.dropout_p = dropout_p

		self.dims = [self.input_dim]
		self.dims.extend(hidden_dims)
		self.dims.append(self.output_dim)

		self.layers = nn.ModuleList([])

		for i in range(len(self.dims)-1):
			ip_dim = self.dims[i]
			op_dim = self.dims[i+1]
			self.layers.append(
				nn.Linear(ip_dim, op_dim, bias=True)
			)

		self.__init_net_weights__()

	def __init_net_weights__(self):

		for m in self.layers:
			m.weight.data.normal_(0.0, 0.1)
			m.bias.data.fill_(0.1)

	def forward(self, x):

		x = x.view(-1, self.input_dim)

		for i, layer in enumerate(self.layers):
			x = layer(x)

			# Do not apply ReLU on the final layer
			if i < (len(self.layers) - 1):
				x = F.relu(x)

			if i < (len(self.layers) - 1):		# No dropout on output layer
				x = F.dropout(x, p=self.dropout_p, training=self.training)

		return x

class CNNet(nn.Module):
	"""
	CNN Network for MNIST classification
	"""

	def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

		super(CNNet, self).__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.dropout_p = dropout_p
		self.pool = nn.MaxPool2d(2, 2)


		#self.conv_dims = [1]
		#self.conv_dims.extend([n_channels])
		#self.conv_output_dim = 
		#self.kernels = kernels
		# Conv layers (no weight initialisation)
		self.conv1 = nn.Conv2d(1, 16, 3)
		self.conv2 = nn.Conv2d(16, 32, 3)
		self.input_dim = 24 * 24 * 32 # Only for MNIST
		# Linear leayers
		self.dims = [self.input_dim]
		self.dims.extend(hidden_dims)
		self.dims.append(self.output_dim)

		self.layers = nn.ModuleList([])

		for i in range(len(self.dims)-1):
			ip_dim = self.dims[i]
			op_dim = self.dims[i+1]
			self.layers.append(
				nn.Linear(ip_dim, op_dim, bias=True)
			)

		self.__init_net_weights__()

	def __init_net_weights__(self):

		for m in self.layers:
			m.weight.data.normal_(0.0, 0.1)
			m.bias.data.fill_(0.1)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		b, c, h, w = x.shape
		x = x.view(-1, c*h*w)

		for i, layer in enumerate(self.layers):
			x = layer(x)

			# Do not apply ReLU on the final layer
			if i < (len(self.layers) - 1):
				x = F.relu(x)

			if i < (len(self.layers) - 1):		# No dropout on output layer
				x = F.dropout(x, p=self.dropout_p, training=self.training)

		return x
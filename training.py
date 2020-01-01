import os
import torch
import warnings
import argparse
import torchvision
import numpy as np
from model import *
from visualize import imshow
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

def main(args = None):
	T = transforms.Compose(
	[transforms.Grayscale(num_output_channels = 1),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transformation

	ip_dim = 96*96 # input dimension
	h1 = int(ip_dim/2) # hidden layer 1 dimension
	op_dim = int(ip_dim/4) # output dimension

	parser = argparse.ArgumentParser(description='Simple training script for training a Vanilla Autoencoder network.')

	parser.add_argument('--data_path', help = 'Path for the downloaded dataset', default = './dataset/')
	parser.add_argument('--dataset', help = 'Dataset name. Must be one of MNIST, STL10, CIFAR10')
	parser.add_argument('--epochs', help = 'Number of epochs', type = int, default = 75)
	parser.add_argument('--batch_size', help = 'Batch size of the data', type = int, default = 28)
	parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 0.001)
	parser.add_argument('--use_cuda', help = 'CUDA usage', type = bool)
	parser.add_argument('--network_type', help='Type of the network layers. Must be one of Conv, FC', default='FC')

	parser = parser.parse_args(args)

	epochs = parser.epochs # number of epochs
	batch_size = parser.batch_size # batch size
	learning_rate = parser.learning_rate # learning rate
	data_path = parser.data_path # path of the dataset

	# Creating dataset path if it doesn't exist
	if parser.data_path is None:
		raise ValueError('Must provide dataset path')
	else:
		data_path = parser.data_path
		if not os.path.isdir(data_path):
    		os.mkdir(data_path)

	# Downloading proper dataset and creating data loader
	if parser.dataset == 'MNIST':
		train_data = torchvision.datasets.MNIST(data_path, train=True, download=True, transforms=T)
		test_data = torchvision.datasets.MNIST(data_path, train=False, download=True, transforms=T)
	elif parser.dataset == 'STL10':
		train_data = torchvision.datasets.STL10(data_path, split='train', download=True, transforms=T)
		test_data = torchvision.datasets.STL10(data_path, split='test', download=True, transforms=T)
	elif parser.dataset == 'CIFAR10':
		train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transforms=T)
		test_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transforms=T)
	elif parser.dataset is None:
		raise ValueError('Must provide dataset name')
	else:
		raise ValueError('Dataset name must be MNIST, STL10 or CIFAR10')
	if parser.data_path is None:
		raise ValueError('Must provide training dataset')

	train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
	test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

	# use CUDA or not
	device = 'cpu'
	if parser.use_cuda == False:
		if torch.cuda.is_available():
			warnings.warn('CUDA is available, please use for faster computation')
		else:
			device = 'cpu'
	else:
		if torch.cuda.is_available():
			device = 'cuda'
		else:
			raise ValueError('CUDA is not available, please set it False')

	# Type of layer
	if parser.network_type == 'FC':
		auto_encoder = model.Autoencoder(ip_dim, h_dim, op_dim).to(device)
	elif parser.network_type == 'Conv':
		auto_encoder = model.ConvolutionAE().to(device)
	else:
		raise ValueError('Network type must be either FC or Conv type')

	# Show some real images
	data_iter = iter(train_loader)
	images, labels = data_iter.next()
	imshow(torchvision.utils.make_grid(images))

	# Train the model
	auto_encoder.train()
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(lr=learning_rate, params=auto_encoder.parameters(), weight_decay=1e-5)

	for n_epoch in range(epochs): # loop over the dataset multiple times
		reconstruction_loss = 0.0
		for i, X, Y in enumerate(train_loader, 0):
			X = X.view(X.size()[0], -1)
			X = Variable(X).to(device)
			Y = Variable(Y).to(device)

			encoded, decoded = auto_encoder(X)

			optimizer.zero_grad()
			loss = criterion(X, decoded)
			loss.backward()
			optimizer.step()

			reconstruction_loss += loss.item()
			if i % 2000 == 1999:
				print('[%d, %5d] Reconstruction loss: %.5f' %
                  (n_epoch+1, i+1, reconstruction_loss/2000))
			reconstruction_loss = 0.0




if __name__ == '__main__':
	main()

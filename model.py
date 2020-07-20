import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

numerical_features = ['boost', 'idf', 'tf', 
		'tfidf', 'total_score', 'category_ss', 
		'feature1_s', 'feature2_ss', 
		'feature3_ss', 'score', 'title']

with open('bert_vocab.txt') as file:
	vocab = file.read().split('\n')

class RNN_block(nn.Module):
	"""
	Model performing embeddings building and 
	recurrent tensor operations.
	Initialize with:

		:param: seq_len - sequence length of 
				word indecies inputs
		:param: vocab_size - number of words in
				the word index
		:param: hidden_size - size of RNN embeddings
		:param: output_size - dimentions of output tensor
		:param: n_layers - number of layers to build RNN
				model with
	"""

	def __init__(self, seq_len, 
		vocab_size, hidden_size=128, 
		output_size=64, n_layers=2):

		super().__init__()
		self.seq_len = seq_len
		self.vocab_size = vocab_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers

		self.embedding = nn.Embedding(
			num_embeddings=self.vocab_size, 
			embedding_dim=3, padding_idx=0)

		self.rnn = nn.RNN(input_size=3, 
						hidden_size=self.hidden_size, 
						num_layers=self.n_layers)
		self.rnn_flat = nn.Flatten()
		self.pool = nn.MaxPool1d(8)
		self.rnn_lin = nn.Linear(
			(self.seq_len*self.hidden_size*self.seq_len)//8, 
			self.output_size)

	def forward(self, x):
		x = x.long().view(-1, self.seq_len)
		x = self.embedding(x)
		x, h_c = self.rnn(x)
		x = self.rnn_flat(x)
		x = x.view(-1, self.seq_len, self.hidden_size*5)
		x = self.pool(x)
		x = self.rnn_flat(x)
		x = F.relu(self.rnn_lin(x))
		return x

class Linear_block(nn.Module):
	"""
	Model performing dense tensors operations.
	Initialize with:

		:param: num_features - number of numeric inputs
		:param: hidden_sizes - iterable of 3 elements
				specifying dimentionality of dense layers
	"""

	def __init__(self, num_features, hidden_sizes=[64, 128, 64]):
		super().__init__()

		self.num_features = num_features

		self.hidden_sizes = hidden_sizes
		self.lin_input = nn.Linear(
			self.num_features, hidden_sizes[0])
		self.lin_hidden = nn.Linear(
			self.hidden_sizes[0], self.hidden_sizes[1])
		self.lin_output = nn.Linear(
			self.hidden_sizes[1], self.hidden_sizes[2])

	def forward(self, x):
		x = x.float().view(-1, self.num_features)
		x = self.lin_input(x)
		x = F.relu(x)
		x = self.lin_hidden(x)
		x = F.relu(x)
		x = self.lin_output(x)
		x = F.relu(x)
		return x


class LTRModel(nn.Module):

	def __init__(self, text_fts, 
		num_fts, 
		hidden_sizes=[64, 128, 64],
		vocab=vocab):
		super().__init__()

		self.text_fts = text_fts
		self.num_fts = num_fts
		self.hidden_sizes = hidden_sizes
		self.vocab = vocab

		self.RNN_block = RNN_block(len(text_fts), 
			len(vocab), hidden_sizes[1], 
			hidden_sizes[-1], 2)

		self.lin_block = Linear_block(
			num_features=len(num_fts))

		self.out = nn.Linear(hidden_sizes[-1]*2, 1)


	def forward(self, x1, x2):
		x1 = self.RNN_block(x1)
		x2 = self.lin_block(x2)
		x = torch.cat([x1, x2], axis=1)
		x = self.out(x)
		return x

class LTRSimpleModel(nn.Module):

	def __init__(self, num_features):
		super().__init__()

		self.num_features = num_features
		self.network = Linear_block(
			num_features=self.num_features)
		self.output = nn.Linear(
			self.network.lin_output.out_features, 1)

	def forward(self, x):
		x = self.network(x)
		return F.relu(self.output(x))
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from tqdm import tqdm

from dataset import QueriesDataset
from model import LTRModel, LTRSimpleModel

num_fts = ['boost', 'idf', 'price', 
		'selling_price_d', 'tf', 
		'tfidf', 'total_score', 
		'clicked']

text_fts = ['bed_size_measure_ss', 
		'bed_type_ss', 'brand_s', 
		'category_ss', 'color_ss', 
		'key_features_s', 
		'material_type_ss', 
		'query_vec', 'style_ss', 
		'title_vec']

target_col = 'score'

torch.manual_seed(66)

def parse_args():
	parser = argparse.ArgumentParser(
		description='Train Valentine`s implimentation of LTR model')

	parser.add_argument(
		'--epochs', '-e', dest='epochs',
		help='Number of epochs to train',
		default=50)

	parser.add_argument(
		'--learning-rate', '-lr', dest='lr',
		help='Leaning rate for optimizer',
		default=0.001)

	parser.add_argument(
		'--weight_decay', '-wd', dest='weight_decay',
		help='Weiht decay for optimizer',
		default=0.01)

	return parser.parse_args()

def get_loaders(df_path, 
	test_size=0.2, batch_size=64):
	"""
	Create train and test datasets and dataloaders.

	:param: df_path - path to *.csv file with
			preloaded information or 
			pandas.DataFrame object if pulled
			from Solr server.
	:param: test_size - fraction of 1 to use
			as test data
	:param: batch_size - number of instances to
			load in minibatch
	:return: tuple where first element
			torch.utils.data.DataLoader object with
			train instances, second element
			torch.utils.data.DataLoader object with
			test instances
	"""

	try:
		df = pd.read_csv(df_path)
	except Exception:
		df = pd.read_csv(df_path, sep='\t')
	finally:
		df = df_path
	dataset = QueriesDataset(
		queries=df['query_text'].tolist(), 
		offline=True, target_col='score')

	train_set, val_set = random_split(dataset, 
		[int((1-test_size)*len(dataset)), 
		int(test_size*len(dataset))])

	train_loader = DataLoader(train_set, 
		batch_size=batch_size)
	val_loader = DataLoader(val_set, 
		batch_size=batch_size)

	return train_loader, val_loader


def mean_absolute_error(y_true, y_pred):
	y_true, y_pred = torch.from_numpy(
		y_true), torch.from_numpy(y_pred)
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def accuracy(outputs, labels):
    _, preds = torch.max(outputs)
    return torch.tensor(
    	torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader, 
	problem_type='binary_classification'):
	"""
	Evaluate performance of the model.

	:param: model - LTRModel object in case of
			binary classification problem,
			LTRSimpleModel object in case of
			regression problem
	:param: val_loader - torch.utils.data.DataLoader object
			with test instances
	:param: problem_type - 'binary_classification' or
			'regression' based on values predicted,
			model and data used
	:return: tuple where first element accuracy of 
			classification for classification problem
			(torch.Tensor object) or MAPE loss otherwise 
			(numpy.array object), second element BCE loss
			(torch.Tensor object) in case of classification
			problem and MSE (mean squared error) otherwise.

	"""

	model.eval()
	acc = []
	loss = []
	for batch in tqdm(val_loader):
		if problem_type == 'binary_classification':
			out = torch.sigmoid(model(batch[0], batch[1]))
			acc.append(accuracy(out, batch[2]))
			loss.append(
				F.binary_cross_entropy(out, batch[2].float()))
		if problem_type == 'regression':
			out = model(batch[1])
			loss.append(F.mse_loss(out, batch[2].float()))
			#print(batch[2].numpy().flatten())
			#print(out.detach().numpy().flatten())
			acc.append(mean_absolute_error(
				batch[2].numpy().flatten().reshape(-1, 1), 
				out.detach().numpy().flatten().reshape(-1, 1)))
	return torch.mean(torch.tensor(acc))*100, torch.mean(torch.tensor(loss))

def train(model, train_loader, val_loader, 
	optimizer, epochs=20, scheduler=None, 
	problem_type='binary_classification'):
	"""
	Train initialized model.

	:param: model - LTRSimpleModel or LTRModel object
	:param: train_loader - torch.utils.data.DataLoader object
			with train instances
	:param: val_loader - torch.utils.data.DataLoader object
			with test instances
	:param: optimizer - optimizer object from torch.optim module
	:param: epochs - number of epochs to train model for
	:param: scheduler - learning rate scheduler object from
			torch.optim.lr_scheduler module
	:param: problem_type - `binary_classiication` for
			classification or `regression` for regression
	:return: LTRSimpleModel or LTRModel object
	"""

	model = model.train()

	for epoch in range(epochs):
		total_loss = 0
		model.zero_grad()
		for batch in tqdm(train_loader):
			x1, x2, y = batch
			if problem_type == 'binary_classification':
				out = torch.sigmoid(model(x1, x2))
				loss = F.binary_cross_entropy(out, y.float())
			if problem_type == 'regression':
				model = model.float()
				out = model(x2)#.float()
				loss = F.mse_loss(out, y.float())
			total_loss += loss.detach()
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
		val_acc, val_loss = evaluate(model, val_loader, problem_type='regression')
		scheduler.step(val_loss)
		rate = scheduler.get_lr()
			
		print(f'epoch: {epoch}, train loss: {total_loss/len(train_loader)}, \
			val loss: {val_loss}, val acc: {val_acc}, lr: nan')
	return model
		

	
def main():

	args = parse_args()
	train_loader, val_loader = get_loaders(
		'catalogsearch_query.csv')

	model = LTRSimpleModel(num_features=len(num_fts))

	optimizer = Adam(model.parameters(),
		lr=args.lr, weight_decay=args.weight_decay)

	scheduler = CyclicLR(optimizer, 
		base_lr=args.lr, max_lr=1e-2, 
		step_size_up=int(len(train_loader.dataset)/2),
		step_size_down=int(len(train_loader.dataset)/2),
		cycle_momentum=False)

	print(scheduler)
	train(model=model, 
		train_loader=train_loader, 
		val_loader=val_loader, 
		optimizer=optimizer, 
		epochs=args.epochs,
		scheduler=scheduler,
		problem_type='regression')



if __name__ == '__main__':
	main()
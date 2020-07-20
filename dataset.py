import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

from build_vocab import build_tokenizer, extend_vocab
from pull_data import pull_data

"""
Note: offline dataset has been collected for
`bed` query only and is used merely for
pipeline buidling
"""



class QueriesDataset(Dataset):
	"""
	Build dataset with information about documents and scores
	Initialize with:

		:param: queries - iterrable with queries to search
		:param: vocab_file - file with words in the vocabulary
		:param: numeric - wheather to use numeric features
				and convert text/categorical to numbers
		:param: target_col - column in pandas.DataFrame
				object containing values to predict.
				For experimental purposes use method
				dummy_targets to generate `clicked`
				target of binary data
		:param: offline - wheather to use online Solr server or
				pull from preloaded frame
		:param: scaling - method used to scale 
				data (not implemented for now)
		:param: offline_file - preloaded *.csv file with 
				data pulled from Solr server
	"""

	def __init__(self, queries, num_docs=10, 
		vocab_file='bert_vocab.txt', 
		numeric=True, offline=False,
		target_col='clicked',
		scaling='max',
		offline_file='sample_1sb.csv'):
		super().__init__()

		self.offline_file = offline_file
		self.queries = [str(x) for x in queries]
		self.num_docs = num_docs
		self.vocab_file = vocab_file
		self.offline = offline
		self.target_col = target_col
		self.scaling = scaling
		self.tokenizer = build_tokenizer(self.vocab_file)
		self.build_set()
		self.dummy_targets()
		#self.scale_features()
		self.numeric = numeric
		if self.numeric:
			self.convert_to_numeric()

	def __len__(self):
		return self.set.shape[0]

	def __getitem__(self, ind):
		numeric_cols = set(self.set.columns).difference(self.text_cols)
		x_text = self.set.loc[ind, self.text_cols].values
		x_text = np.vstack(x_text)
		if isinstance(ind, int):
			x_numeric = self.set.loc[ind, numeric_cols].drop(
				[self.target_col, 'Unnamed: 0']).values
		else:
			x_numeric = self.set.loc[ind, numeric_cols].drop(
				[self.target_col, 'Unnamed: 0'], axis=1).values
		x_numeric = np.hstack(x_numeric)
		y = self.set.loc[ind, self.target_col] #.get_dummies()
		x_text = torch.from_numpy(x_text)
		x_numeric = torch.from_numpy(x_numeric)
		y = torch.from_numpy(np.array([y]))
		return x_text, x_numeric, y

	def convert_to_numeric(self):
		"""
		Replace text features with sequences of numbers based
		on number in keras.Tokenizer().word_index. Fields
		*_s and *_ss are filled with `0|Unknown` if values
		are missing and converted to sequences of word indecies.
		Missing numeric values are replaced with -99999.
		Set sequences are padded with static method pad_sequence
		"""
		self.text_cols = ['ft2_vec', 'ft3_vec']
		for col in self.set.columns:
			if col.endswith('_ss') or col.endswith('_s'):
				self.set.loc[:, col] = self.set[col].fillna('0|Unknown')
				self.set.loc[:, col] = self.set[col].apply(
					lambda x: x.split('|')[-1])
				self.set.loc[:, col] = self.set[col].apply(
					lambda x: self.pad_sequence(self.encode_text([x])))
				self.text_cols.append(col)
			else:
				self.set.loc[:, col] = self.set[col].fillna(-99999)
		self.set.drop(['feature2_ss', 'feature3_ss', 'feature1_s', 'id'], 
			axis=1, inplace=True)

	@staticmethod
	def pad_sequence(seq, length=5):
		"""
		Pad or truncate sequence to the constant value.

		:param: seq - sequence to pad or truncate
		:param: length - total number of elements in
				the sequence

		:return: the same array if it contain `length` elements
				truncated sequence if number of elements greter 
				then `length`
				padded with `0` array otherwise
		"""
		if len(seq) == length:
			return np.array(seq).astype(np.int64)
		elif len(seq) > 5:
			return np.array(seq[:5]).astype(np.int64)
		else:
			arr = np.zeros(length)
			arr[:len(seq)] = seq
			
			return arr.astype(np.int64)

	def scale_features(self, inverse=False):
		"""
		Transform fetures
		:param: inverse - if False scale fetures 
				with strategy dividing by maximum 
				value as default, unscale by 
				multiplication by maximum otherwise 
		"""
		self.maxs = []
		for col in self.set.columns:
			if self.set[col].dtype == 'object':
				continue
			m = self.set[col].max()
			self.set.loc[:, col] = self.set[col]/m
			self.maxs.append(m)

	def encode_text(self, text):
		return self.tokenizer.texts_to_sequences(text)[0]

	def decode_text(self, ids):
		return self.tokenizer.sequences_to_texts(ids)

	def build_set(self):
		"""
		Build a dataset to train model on.
		If attribute offile is True build from
		preloaded dataset, build from Solr 
		server otherwise
		"""

		self.set = pd.DataFrame()
		self.tokenizer.fit_on_texts(self.queries)
		print('Building dataset')
		if not self.offline:
			for q in tqdm(self.queries):
				d = pull_data(query=q, 
					rows=self.num_docs, 
					verbose=False)
				for i in d.index:
					dict_to_append = d.loc[i, :].to_dict()
					dict_to_append['feature3_ss'] = q
					print(q)
					q_seq = self.encode_text(
						[str(dict_to_append['feature3_ss'])])
					t_seq = self.encode_text(
						[str(dict_to_append['feature2_ss'])])
					dict_to_append['ft3_vec'] = self.pad_sequence(q_seq)
					dict_to_append['ft2_vec'] = self.pad_sequence(t_seq)
					self.set = self.set.append(
						dict_to_append, ignore_index=True)
		else:
			self.set = pd.read_csv(self.offline_file)
			self.set['feature3_ss'] = 'fancy bed'
			self.set['ft3_vec'] = self.set['feature3_ss'].astype(
				str).apply(self.encode_text)
			self.set['ft3_vec'] = self.set['ft3_vec'].apply(
				self.pad_sequence)
			self.set['ft2_vec'] = self.set['feature2_ss'].astype(
				str).apply(self.encode_text)
			self.set['ft2_vec'] = self.set['ft2_vec'].apply(
				self.pad_sequence)

	def dummy_targets(self):
		"""
		Generates targets not present in the dataset.
		As default value binary data `clicked` is chosen.
		"""

		if 'clicked' not in self.set.columns:

			self.set['clicked'] = np.random.choice([1, 0], 
									size=self.set.shape[0], 
									p=[0.5, 0.5])
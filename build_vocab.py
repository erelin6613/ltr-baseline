import pandas as pd
import time
import json

from pull_data import pull_data
from keras_tokenizer import Tokenizer, tokenizer_from_json

queries_file = 'catalogsearch_query.csv'
text_col = 'query_text'

replace_dict = {
	'w/': 'with ',
	'+': ' ',
	'/': ' ',
	'soda': 'sofa',
	'.': ' '
}

def clean_text(string):
	if isinstance(string, str) == False:
		string = str(string)
	if string.replace(' ', '').isdigit():
		return ''
	for k, v in replace_dict.items():
		string = string.replace(k, v)
	return string

def dummy_tokenize(text):
	tokens = set(text.split(' '))
	return tokens

def build_vocab(file=queries_file,
	text_cols=[text_col], 
	tokenizer=None,
	write_vocab=False):

	try:
		df = pd.read_csv(file)
	except Exception:
		df = pd.read_csv(file, sep='\t')
	if tokenizer is None:
		tokenizer = Tokenizer(oov_token='<OOV>', lower=False)
	if 'num_results' in df.columns:
		df = df.loc[df['num_results'] != 0]

	for col in text_cols:
		df.loc[:, col] = df[col].apply(clean_text)
		tokenizer.fit_on_texts(df[col])
	print('Total words', len(tokenizer.word_index))
	if write_vocab:
		with open('vocab.txt', 'a') as f:
			for each in tokenizer.word_index.keys():
				f.write(each+'\n')

	return tokenizer

def build_tokenizer(vocab_file='./bert_vocab.txt'):

	with open(vocab_file) as f:
		lines = f.readlines()
	lines = [x.strip() for x in lines]
	tokenizer = Tokenizer(oov_token=1, num_words=len(lines)+1)
	tokenizer.fit_on_texts(lines)
	return tokenizer


def extend_vocab(text=None, vocab_file='./bert_vocab.txt'):
	with open(vocab_file) as f:
		vocab = f.read().split('\n')

	words = [x for x in clean_text(text).split(' ')]
	for word in words:
		if word not in vocab:
			vocab.append(word)
			print(f'{word} added to vocabulary')
	with open(vocab_file, 'w') as file:
		file.write('\n'.join(vocab)[1:])

# extend_vocab('bed')

"""
tokenizer = build_tokenizer()

df = pd.read_csv('catalogsearch_query.csv', sep='\t')

df.loc[:, 'query_text'] = df['query_text'].apply(str)
extend_vocab(' '.join(df['query_text'].tolist()))
"""
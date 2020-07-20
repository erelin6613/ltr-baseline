import argparse
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm
import requests
import json

needed_fields = ['debug', 'id', 
		'feature1_s', 'feature2_ss', 
		'feature3_ss']

parse_scores = {
	'tfidf': 'boost * idf * tf',
	'idf': 'idf, computed as log',
	'tf': 'tf, computed as freq',
	'boost': 'boost'
		}

def parse_args():
	parser = argparse.ArgumentParser(
		description='Train Valentine`s implimentation of LTR model')

	parser.add_argument(
		'--base_url', '-bu', dest='base_url',
		help='Url of Solr home server, f.e. http://solrhost.com/solr/',
		required=True)

	parser.add_argument(
		'--start', '-s', dest='start',
		help='Index from which to start pull',
		default=0)

	parser.add_argument(
		'--rows', '-r', dest='rows',
		help='Number of records to pull',
		default=1000)

	parser.add_argument(
		'--query', '-q', dest='query',
		help='Search query',
		default='bed')

	parser.add_argument(
		'--write_name', '-wn', dest='write_name',
		help='File name to write pulled data to',
		default=None)

	return parser.parse_args()

def get_debug_info(debug_dict):
	"""
	Pull debug information to extract numeric scores
	and features calculated by Solr

	:param: debug_dict - dictionary-like object
			accessed from Solr query (inside the
			value ['debug']['explain'] or the response

	:return: info - pandas.DataFrame object with 
				extracted information

	"""

	info = pd.DataFrame()

	for doc in list(debug_dict.keys()):
		s = {}
		s['doc_id'] = doc
		s['total_score'] = debug_dict[doc]['value']
		for k in debug_dict[doc]['details']:
			s['tfidf'] = k['value']
			for each in k['details']:
				for key in parse_scores.keys():
					if parse_scores[key] in each['feature2_s']:
						s[key] = each['value']
		info = info.append(s, ignore_index=True)

	return info

def get_features(docs_list, debug_df, features=needed_fields):
	"""
	Extract features from Solr response and extend debug information.

	:param: docs_list - interable with information
			accessed by ['response']['docs'] from
			Solr response
	:param: debug_df - pandas.DataFrame object containing
			scoring metrics
	:param: features - features to extract about products
			based on doc_id

	:return: debug_df - extended pandas.DataFrame object
			with debug and extracted fields as specified
			by features parameter
	"""

	for doc in docs_list:
		ind = debug_df.loc[debug_df['feature1_s']==doc['id']].index
		#print(doc)
		for f in needed_fields:
			if f == 'debug':
				continue
			try:
				if isinstance(doc[f], list):
					debug_df.loc[ind, f] = doc[f][0]
				else:
					debug_df.loc[ind, f] = doc[f]
			except Exception as e:
				debug_df.loc[ind, f] = None

	return debug_df



def pull_data(base_url='http://solrhost.com/solr/', 
	start=0, rows=100, query='', fl=needed_fields, 
	verbose=True, collection='collection'):
	"""
	Get data from Solr server containing regular information
	and numerical metrics explaining the ranking

	:param: base_url - url of Solr home
	:param: start - index from which to start pull
	:param: rows - number of records to pull
	:param: query - search query
	:param: fl - fields to extract from server about documents
	:param: verbose - wheather to log loading progress

	:return: pandas.DataFrame object with debug and regular
			information 
	"""

	if len(fl) > 1:
		fields = '%2C%20'.join(fl)
	else:
		fields = fl[0]
	
	url = base_url+'{collection}/select?debug.explain.structured=true\
		&debugQuery=on&fl={}&q={}&rows={}&start={}'.format(
			fields, query, rows, start)
	#print(url)
	if verbose:
		print('loading data...')
	r = requests.get(url)
	debug = json.loads(r.text)['debug']['explain']
	docs = json.loads(r.text)['response']['docs']

	s = get_debug_info(debug)
	data = get_features(docs, s)
	if verbose:
		print('Done!')
	return data

if __name__ == '__main__':
	args = parse_args()
	df = pull_data(
		base_url=args.base_url,
		start=args.start,
		rows=args.rows,
		query=args.query)
	if args.write_name is not None:
		df.to_csv(args.write_name+'.csv')

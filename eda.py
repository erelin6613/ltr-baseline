import os
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
import lxml.html
import re
import csv

from pull_data import pull_data
from products import *

"""
This script is a helper to gain some insights
about the data we have and has a few functions
to convert dataframe to xml format suitable
for solr.
"""

def explore_products():

	df = pull_data()
	print(df.isna().sum())

def split_attrs(df, attr, drop_original=True):
	for i in range(5):
		try:
			df['{}_{}'.format(attr, i)] = df.loc[i, attr].split(',')[i]
		except Exception:
			df['{}_{}'.format(attr, i)] = None
	if drop_original:
		df.drop(attr, axis=1, inplace=True)
	return df

def create_cats(df, col='categories'):
	cats = set()
	for each in df[col]:
		for c in each.split(','):
			cats.add(c)
	# print(len(cats))
	return cats


def get_raw_text(df):
	cols = ['title', 'title_synonyms', 'description',
		'short_description']
	for col in cols:
		if col == 'title_synonyms':
			df.loc[:, col] = df[col].apply(lambda x: x[0])
			continue
		df.loc[:, col] = df[col].apply(clean_up_text)
	return df[cols]

def clean_up_text(text):

	soup = BeautifulSoup(str(text), 'lxml')
	text = ''
	for el in soup:
		text += el.get_text()
	text = re.sub(r'[0-9]*\&quot;', r'', 
		lxml.html.document_fromstring(text).text_content(), 
		flags=re.MULTILINE)
	return text

def csv_converter(csv_file):

	def csv_to_xml(csv_file):         
		f = open(csv_file)
		csv_f = csv.reader(f)   
		data = []

		for row in csv_f: 
		   data.append(row)
		f.close()
		return data

	def clean_string(string):
		#return string
		try:
			return string.replace('&', 'and')
		except Exception:
			return string #.replace('&', 'and')

	def convert_row(row):
		string = """<doc>
		<field name='id'>%s</field>
		<field name='brand' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='colors' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='dimension' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='manufacturer' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='name' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='weight' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='category' type='TextField' indexed='true' stored='true'>%s</field>
		<field name='category_1' type='TextField' indexed='true' stored='true'>%s</field>
		</doc>""" % (row[0], row[1], row[2], 
			row[3], row[4], row[5], row[6],
			row[7], row[8])
		cs = clean_string(string)
		#print(cs)
		return string

	data = csv_to_xml(csv_file)
	xml_name = csv_file.split('.')[0]+'.xml'
	with open(xml_name, 'a') as f:
		f.write('<add>')
		for each in data:
			f.write(convert_row(each))
		f.write('</add>')
	print('done!')

if __name__ == '__main__':
	explore_products()
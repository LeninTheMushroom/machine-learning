
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import string
from gensim.models import Doc2Vec as Doc2Vec
from gensim.models import KeyedVectors as KeyedVectors
from nltk.stem.snowball import SnowballStemmer
import csv
from nltk.corpus import stopwords
import io
from multiprocessing import Process, Queue
import time
import csv


# In[3]:

data_dir = '/home/uasa/Desktop/data/'
google_vec_filepath = data_dir + 'GoogleNews-vectors-negative300.bin'
lexvec_filepath = data_dir + 'lexvec.enwiki+newscrawl.300d.W+C.pos.vectors'
quora_train_filepath = data_dir + 'train.csv'
quora_test_filepath = data_dir + 'test.csv'


class MultiprocStringParser(object):

	trantab = string.maketrans(",.?/", "    ")
	unwanted = '\"#$%^&()*+-_:;<=>@[\\]{|}~\'1234567890'
	delimiter = ',\"'
	#delimiter = '\t'
	test_format = {'f':1, 's':2, 't':0, 'l':3}
	train_format = {'f':3, 's':4, 't':5, 'l':6}
	StopWords = stopwords.words("english")
	stemmer = SnowballStemmer("english")

	def __init__(self, test=True):
		print('Creating parser - loading vector files at')
		start = time.time()
		print(lexvec_filepath)
		self.word_vectors = KeyedVectors.load_word2vec_format(lexvec_filepath)
		if(test == True):
			self.params = self.test_format
		else:
			self.params = self.train_format
		print(str(time.time()-start) + ' seconds to load')

	def check_vocab(self, x, y):
		if x in self.word_vectors.vocab:
			if y in self.word_vectors.vocab:
				return self.word_vectors.similarity(x, y)
		return 0

	def stem(self, word):
		return self.stemmer.stem(unicode(word, errors='ignore'))

	def get_word2vec_score(self, ser1, ser2):
		score = 0.0
		max_score = 0.0
		for word1 in ser1:
			for word2 in ser2:
				sim = self.check_vocab(word1, word2)
				score+= sim
				if(sim > max_score):
					max_score = sim
		return score, max_score

	def clean(self, lines, q):
		result = []
		length = self.params['l']
		first = self.params['f']
		second = self.params['s']
		third = self.params['t']
		for line in lines:
			questions = line[:-2].split(self.delimiter)
			if(len(questions) >= length):
				# discard unwanted characters
				f = questions[first].translate(self.trantab, self.unwanted).lower()
				s = questions[second].translate(self.trantab, self.unwanted).lower()
				t = questions[third]
				# discard unwanted words
				f = ' '.join([word for word in f.split() if word not in self.StopWords])
				s = ' '.join([word for word in s.split() if word not in self.StopWords])
				# build differences
				l1 = f.split()
				l2 = s.split()
				difference1 = set(l1).difference(l2)
				difference2 = set(l2).difference(l1)
				common = set(l1).intersection(l2)
				# mark similiar words as common
				for word1 in difference1.copy():
					for word2 in difference2.copy():
						 if (self.stem(word1)==self.stem(word2)):
							common.add(word1)
							difference1.discard(word1)
							difference2.discard(word2)                          
					
				text_difference1 = ' '.join([word for word in difference1])
				text_difference2 = ' '.join([word for word in difference2])
				text_common = ' '.join([word for word in common])
				#result
				message1 = [f, s, t, len(l1), len(l2),
				 text_difference1, len(difference1),
				  text_difference2, len(difference2),
				   text_common, len(common)]
				# count word2vec
				dif_score = self.get_word2vec_score(difference1, difference2)
				d1c_score = self.get_word2vec_score(difference1, common)
				d2c_score = self.get_word2vec_score(difference2, common)
				# write resulting feature vector
				message2 = [dif_score[0], dif_score[1],
				 d1c_score[0], d1c_score[1],
				  d2c_score[0], d2c_score[1]]
				message = []

				message.extend(message1)
				message.extend(message2)
				result.append(message)
		q.put(result)


def get_file_format(file_name):
	resulting_filepath = data_dir + file_name
	if('test' in file_name):
		return (True, resulting_filepath)
	elif ('train' in file_name):
		return (False, resulting_filepath)
	else:
		print('non-specified format - only \'train\' and \'test\' currently supported (add to filename)')
		return None

def divide_text_to_chunks(file_name, ch_size):
	form = get_file_format(file_name)
	current = 0
	chunk = []
	resulting_list = []
	k = 0
	with open(form[1], 'r') as f:
		next(f)
		for line in f:
			chunk.append(line)
			current+=1
			if(current >=ch_size):
				resulting_list.append(chunk)
				chunk = []
				current = 0
		resulting_list.append(chunk)
	return form[0], resulting_list

if __name__ == '__main__':
	print('Text Feature Construction')
	print('160 seconds for text vector load (2.2 Gb)')
	print('5000 file lines per second on FX-8320 without word2vec evaluation')
	print('3300 lines per second on FX-8320 with word2vec evaluation')

	q = Queue()
	jobs = []
	
	parser_format, chunks = divide_text_to_chunks('test.csv', 294000) # length is 293226*8 for test and 50538*8 for train
	parser = MultiprocStringParser(parser_format)
	sizes = []
	print('got chunks with sizes:')
	for ch in chunks:
		sizes.append(len(ch))
	print(sizes)
	start_time = time.time()
	for x in range (len(chunks)):
		p = Process(target=parser.clean, args=(chunks[x],q))
		jobs.append(p)
		p.start()
	
	result_list = []
	for i in range(len(chunks)):
		result_list.extend(q.get())
	for job in jobs:
		job.join()

	print(time.time() - start_time)

	myfile = open(data_dir + 'multithread-test.csv', 'wb')
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
	for item in result_list:
		wr.writerow(item)
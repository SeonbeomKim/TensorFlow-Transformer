#https://arxiv.org/pdf/1508.07909.pdf Byte-Pair Encoding (BPE)

import re, collections
import numpy as np # 1.15
import pandas as pd # 0.23


def read_pd_format(np_format):
	df = pd.DataFrame(np_format, index=np.arange(len(np_format))),
	return df

def make_bpe_format(path):
	space_symbol = '</w>'
	pad_symbol = '</p>'

	data = pd.read_csv(path, sep=" ", header=None) # pad value is NaN
	data.fillna(pad_symbol, inplace=True) # NaN => pad_symbol

	# to numpy
	data = np.array(data)
	word2idx = {}
	for i, row in enumerate(data):
		for j, word in enumerate(row):
			if word != pad_symbol: # pad는 처리 안함.
				# abc => a b c space_symbol
				data[i][j] = ' '.join(list(word)) + ' ' + space_symbol
				
				# word frequency
				if data[i][j] in word2idx:
					word2idx[data[i][j]] += 1
				else:
					word2idx[data[i][j]] = 1


	for k in word2idx:
		print(k, word2idx[k])
	print(data)
	# 눈으로 쉽게 확인하기위해 짠 테스트코드
	#data = read_pd_format(data)

	print(data)
	#print(len(data[0]))
	#print(np.array(data))
	#print(np.array(data).shape)
		


def get_stats(vocab):
	pairs = collections.defaultdict(int)
	for word, freq in vocab.items():
		#print(word, freq)
		symbols = word.split()
		for i in range(len(symbols)-1):
			pairs[symbols[i],symbols[i+1]] += freq
	return pairs

def merge_vocab(pair, v_in):
	v_out = {}
	bigram = re.escape(' '.join(pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in v_in:
		w_out = p.sub(''.join(pair), word)
		v_out[w_out] = v_in[word]
	return v_out

def bpe_testcode():
	#vocab = {'l o w </w>' : 1, 'l o w e r </w>' : 1, 'n e w e s t </w>':1, 'w i d e s t </w>':1}
	vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2, 'n e w e s t </w>':6, 'w i d e s t </w>':3}
	num_merges = 12

	for i in range(num_merges):
		pairs = get_stats(vocab)
		best = max(pairs)
		vocab = merge_vocab(best, vocab)
	print(vocab)

def make_bpe_format_testcode():
	path = "./testdata.en"
	make_bpe_format(path)

#bpe_testcode()
make_bpe_format_testcode()

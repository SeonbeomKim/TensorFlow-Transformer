#-*-coding:utf-8

# https://arxiv.org/abs/1508.07909 Byte-Pair Encoding (BPE)
# https://lovit.github.io/nlp/2018/04/02/wpm/ 참고

import re, collections
import numpy as np # 1.15
import pandas as pd # 0.23


# word:"abc" => "a b c space_symbol"
def word_split_for_bpe(word, space_symbol='</w>'):
	return ' '.join(list(word)) + ' ' + space_symbol


# chunksize씩 처리하도록 구현하자.
def read_document(path):
	document = []

	with open(path, 'r', encoding='utf-8') as f:
		for i, sentence in enumerate(f):
			if sentence == '\n' or sentence == ' ' or sentence == '':
				break
			document.append(sentence.split())
	return document


# word frequency 추출.
def get_word_frequency_dict_for_bpe_from_document(path_list, space_symbol='</w>', except_symbol={}, top_k=None):
	word_frequency_dict = {}

	for path in path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for i, sentence in enumerate(f):
				if sentence == '\n' or sentence == ' ' or sentence == '':
					break
				
				for word in sentence.split():
					# &apos; 같은 단어를 '로 바꾸는 등의 치환.
					if word in except_symbol:
						word = except_symbol[word]
					
					split_word = word_split_for_bpe(word, space_symbol)
					
					# word frequency
					if split_word in word_frequency_dict:
						word_frequency_dict[split_word] += 1
					else:
						word_frequency_dict[split_word] = 1

	if top_k is None:
		return word_frequency_dict
	
	else:
		# top_k frequency word
		sorted_word_frequency_list = sorted(
					word_frequency_dict.items(), # ('key', value) pair
					key=lambda x:x[1], # x: ('key', value), and x[1]: value
					reverse=True
				) # [('a', 3), ('b', 2), ... ] 
		top_k_word_frequency_dict = dict(sorted_word_frequency_list[:top_k])
	
		return top_k_word_frequency_dict


def merge_dictionary(dic_a, dic_b):
	for i in dic_b:
		if i in dic_a:
			dic_a[i] += dic_b[i]
		else:
			dic_a[i] = dic_b[i]
	return dic_a


# 문서를 읽고, bpe 적용. cache 사용할것.
def document_preprocess_for_bpe(path, space_symbol='</w>', pad_symbol='</p>', eos_symbol='</e>', except_symbol={}, merge_info=None):
		
	bpe = []
	with open(path, 'r', encoding='utf-8') as f:
		for i, sentence in enumerate(f):
			print(i)
			row = []
			if sentence == '\n' or sentence == ' ' or sentence == '':
				break

			for word in sentence.split():
				# &apos; 같은 단어를 '로 바꾸는 등의 치환.
				if word in except_symbol:
					word = except_symbol[word]

				# "abc" => "a b c space_symbol"
				split_word = word_split_for_bpe(word, space_symbol)
				
				# merge_info를 이용해서 merge.  "a b c space_symbol" ==> "ab cspace_symbol"
				merge = merge_a_word(merge_info, split_word)

				# 안합쳐진 부분은 다른 단어로 인식해서 공백기준 split 처리해서 sentence에 extend
				row.extend(merge.split())
		
			# eos 추가.
			row.append(eos_symbol)
			bpe.append(row)
			
	return bpe



# 2-gram frequency table 추출.
def get_stats(word_frequency_dict):
	# word_frequency_dict: dictionary
	pairs = collections.defaultdict(int) # tuple form으로 key 사용 가능함.
	for word, freq in word_frequency_dict.items():
		symbols = word.split()
		for i in range(len(symbols)-1):
			pairs[symbols[i],symbols[i+1]] += freq
	
	# tuple을 담고 있는 dictionary 리턴.
	return pairs 

# pairs 중에서 가장 높은 frequency를 갖는 key 리턴.
def check_merge_info(pairs):
	best = max(pairs, key=pairs.get)
	return best

# frequency가 가장 높은 best_pair 정보를 이용해서 단어를 merge.
def merge_word2idx(best_pair, word_frequency_dict):
	# best_pair: tuple ('r','</w>')
	# word_frequency_dict: dictionary
	
	v_out = collections.OrderedDict() # 입력 순서 유지

	bigram = re.escape(' '.join(best_pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in word_frequency_dict:
		# 만약 ''.join(best_pair): r</w> 이고, word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.
		w_out = p.sub(''.join(best_pair), word)
		v_out[w_out] = word_frequency_dict[word]
	return v_out


# from bpe to idx
def make_bpe2idx(word_frequency_dict):
	bpe2idx = {
				'UNK':0,
				'</g>':1, #go
				'</e>':2, #eos
				'</p>':3
			}
	idx = 4
	
	idx2bpe = {
				0:'UNK',
				1:'</g>', #go
				2:'</e>', #eos
				3:'</p>'
			}
	
	for word in word_frequency_dict:
		for bpe in word.split():
			# bpe가 bpe2idx에 없는 경우만 idx 부여.
			if bpe not in bpe2idx:
				bpe2idx[bpe] = idx
				idx2bpe[idx] = bpe
				idx += 1
	return bpe2idx, idx2bpe



def get_bpe_information(word_frequency_dict, num_merges=10):
	#word_frequency_dict = {'l o w </w>' : 1, 'l o w e r </w>' : 1, 'n e w e s t </w>':1, 'w i d e s t </w>':1}
	
	#merge_info: 합친 정보를 기억하고있다가. 나중에 데이터를 같은 순서로만 합치면 똑같이 됨.
	merge_info = collections.OrderedDict() # 입력 순서 유지
	
	word_frequency_dict = collections.OrderedDict(word_frequency_dict) # 입력 순서 유지(cache 구할 때 순서 맞추려고)
	cache = word_frequency_dict.copy() # 나중에 word -> bpe 처리 할 때, 빠르게 하기 위함.

	import time
	start = time.time()

	log = 1 # 1000등분마다 찍을것.
	for i in range(num_merges):
		#1000등분마다 로그
		if i % (num_merges / 1000) == 0:
			print( log,'/',1000 , 'time:', time.time()-start )
			log += 1 # 1000등분마다 찍을것.

		pairs = get_stats(word_frequency_dict) # 2gram별 빈도수 추출.
		best = check_merge_info(pairs) # 가장 높은 빈도의 2gram 선정
		word_frequency_dict = merge_word2idx(best, word_frequency_dict) # 가장 높은 빈도의 2gram을 합침.

		#merge 하는데 사용된 정보 저장.
		merge_info[best] = i 
	

	# 빠른 변환을 위한 cache 저장. 기존 word를 key로, bpe 결과를 value로.
	merged_keys = list(word_frequency_dict.keys())
	for i, key in enumerate(cache):
		cache[key] = merged_keys[i]

	# voca 추출.
	bpe2idx, idx2bpe = make_bpe2idx(word_frequency_dict)
	return bpe2idx, idx2bpe, merge_info, cache



def merge_a_word(merge_info, word):
	# merge_info: OrderedDict {('s','</w>'):0, ('e', '</w>'):1 ... }
	# word: "c e m e n t </w>" => "ce m e n t<\w>" 되어야 함.
	
	for info in merge_info:
		bigram = re.escape(' '.join(info))
		p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

		# 만약 ''.join(info): r</w> 이고, word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.
		word = p.sub(''.join(info), word)
	return word



def save_dictionary(path, dictionary):
	np.save(path, dictionary)

def load_dictionary(path, encoding='utf-8'):
	data = np.load(path, encoding='bytes').item()
	return data


def testcode_3():
	except_symbol = {'&apos;':"""'""", '@-@': '-', '&quot;':'''"''', '&amp;':'&'}

	'''
	path_en = "../dataset/corpus.tc.de/corpus.tc.de"
	path_de = "../dataset/corpus.tc.en/corpus.tc.en"
	
	en_word_frequency_dict = get_word_frequency_dict_for_bpe_from_document(
				path_list=[path_en], 
				space_symbol='</w>', 
				except_symbol=except_symbol,
				top_k=100000#None
			) #ok
	de_word_frequency_dict = get_word_frequency_dict_for_bpe_from_document(
				path_list=[path_de], 
				space_symbol='</w>', 
				except_symbol=except_symbol,
				top_k=100000#None
			)
	
	merge_dict = merge_dictionary(en_word_frequency_dict, de_word_frequency_dict)
	np.save('./new_merge_dictionary.npy', merge_dict)
	'''
	word_frequency_dict = load_dictionary('./new_merge_dictionary.npy')
	print(len(word_frequency_dict))
	
	#import time
	#start = time.time()
	
	# 1번에 1.09초 
	bpe2idx, idx2bpe, merge_info, cache = get_bpe_information(word_frequency_dict, num_merges=37000)#100
	#print(time.time()-start)
	
	print(len(bpe2idx))
	for i, key in enumerate(cache):
		print(key, cache[key])
		if i == 5:
			break
	
	save_dictionary('./new_bpe2idx.npy', bpe2idx)
	save_dictionary('./new_idx2bpe.npy', idx2bpe)
	save_dictionary('./new_merge_info.npy', merge_info)
	save_dictionary('./new_cache.npy', cache)

	#bpe2idx = load_dictionary('./bpe2idx.npy')
	#merge_info = load_dictionary('./merge_info.npy', encoding='utf-8')
	#start = time.time()
	#test = merge_a_word(merge_info, "t e s t </w>")
	#print(time.time()-start)
	#print(test)

	'''

	path_en = "./testdata.en"
	en_word_frequency_dict = get_word_frequency_dict_for_bpe_from_document(
				path_list=[path_en], 
				space_symbol='</w>', 
				except_symbol=except_symbol,
				top_k=50000#None
			) #ok
	bpe2idx, idx2bpe, merge_info, cache = get_bpe_information(en_word_frequency_dict, num_merges=100)#100
	for i in merge_info:
		print(i, merge_info[i])
	'''

testcode_3()
# https://arxiv.org/abs/1508.07909 Byte-Pair Encoding (BPE)
# https://lovit.github.io/nlp/2018/04/02/wpm/ 참고

import re, collections
import numpy as np # 1.15
import pandas as pd # 0.23

# 눈으로 쉽게 확인하기위해 짠 테스트코드import pandas as pd # 0.23
#data = read_pd_format(data)
def read_pd_format(np_format):
	df = pd.DataFrame(np_format, index=np.arange(len(np_format))),
	return df


# word:"abc" => "a b c space_symbol"
def word_split_for_bpe(word, space_symbol='</w>'):
	return ' '.join(list(word)) + ' ' + space_symbol


# document read and padding.
def read_document_and_pad(path, pad_symbol='</p>'):
	data = pd.read_csv(path, sep=" ", header=None) # pad value is NaN
	data.fillna(pad_symbol, inplace=True) # NaN => pad_symbol	
	return np.array(data)


# 문서를 읽고, bpe 적용.
def document_preprocess_for_bpe(path, space_symbol='</w>', pad_symbol='</p>', eos_symbol='</e>', merge_info=None):
	data = read_document_and_pad(path, pad_symbol) #numpy type
	
	bpe = []
	for i, row in enumerate(data):
		sentence = []
		for j, word in enumerate(row):
			if word != pad_symbol: # pad는 처리 안함.
				# "abc" => "a b c space_symbol"
				split_word = word_split_for_bpe(word, space_symbol)
				
				# merge_info를 이용해서 merge.  "a b c space_symbol" ==> "ab cspace_symbol"
				merge = merge_a_word(merge_info, split_word)

				# 안합쳐진 부분은 다른 단어로 인식해서 공백기준 split 처리해서 sentence에 extend
				sentence.extend(merge.split())

		# eos 추가.
		sentence.append(eos_symbol)
		bpe.append(sentence)
		
	return bpe



# 문서 읽어서 bpe를 적용할 수 있는 format으로 변환. 
def get_word_frequency_dict_for_bpe_from_document(path, space_symbol='</w>', pad_symbol='</p>'):
	data = read_document_and_pad(path, pad_symbol) #numpy type
	word_frequency_dict = {}

	for i, row in enumerate(data):
		for j, word in enumerate(row):
			if word != pad_symbol: # pad는 처리 안함.
				# "abc" => "a b c space_symbol"
				data[i][j] = word_split_for_bpe(word, space_symbol)
				
				# word frequency
				if data[i][j] in word_frequency_dict:
					word_frequency_dict[data[i][j]] += 1
				else:
					word_frequency_dict[data[i][j]] = 1

	return word_frequency_dict


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


# frequency가 가장 높은 best_pair 정보를 이용해서 단어를 merge.
def merge_word2idx(best_pair, word_frequency_dict):
	# best_pair: tuple ('r','</w>')
	# word_frequency_dict: dictionary

	v_out = {}
	bigram = re.escape(' '.join(best_pair))
	p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
	for word in word_frequency_dict:
		# 만약 ''.join(best_pair): r</w> 이고, word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.
		w_out = p.sub(''.join(best_pair), word)
		v_out[w_out] = word_frequency_dict[word]
	return v_out


# pairs 중에서 가장 높은 frequency를 갖는 key 리턴.
def check_merge_info(pairs):
	best = max(pairs, key=pairs.get)
	return best


# from bpe to idx
def make_bpe2idx(word_frequency_dict):
	bpe2idx = {
				'UNK':0,
				'</g>':1, #go
				'</e>':2, #eos
				'</p>':3
			}
	idx = 4
	for word in word_frequency_dict:
		for bpe in word.split():
			# bpe가 bpe2idx에 없는 경우만 idx 부여.
			if bpe not in bpe2idx:
				bpe2idx[bpe] = idx
				idx += 1
	return bpe2idx



def get_bpe_information(word_frequency_dict, num_merges=10):
	#word_frequency_dict = {'l o w </w>' : 1, 'l o w e r </w>' : 1, 'n e w e s t </w>':1, 'w i d e s t </w>':1}
	
	#merge_info: 합친 정보를 기억하고있다가. 나중에 데이터를 같은 순서로만 합치면 똑같이 됨.
	merge_info = collections.OrderedDict() # 입력 순서 유지
	recover_info = collections.OrderedDict()

	i = 0 #iter
	while True:
		pairs = get_stats(word_frequency_dict) # 2gram별 빈도수 추출.
		best = check_merge_info(pairs) # 가장 높은 빈도의 2gram 선정
		word_frequency_dict = merge_word2idx(best, word_frequency_dict) # 가장 높은 빈도의 2gram을 합침.
		
		# 문서 전처리용 정보 추출. merge_info에 저장된 순으로 합치면 됨.
		if i < num_merges:
			merge_info[best] = i #merge 하는데 사용된 정보 저장.
			i += 1

		# 추 후 딥러닝에서 번역한 후에 subword들을 합칠 때 사용되는 정보.
		# len(pairs)가 1일 때 까지 수행. 그 다음 iter에서는 0이 되어서 오류나므로 break.
		if len(pairs) >= 1: 
			recover_info[best] = i
			if len(pairs) == 1:
				break
	
	bpe2idx = make_bpe2idx(word_frequency_dict)
	return bpe2idx, merge_info, recover_info


def merge_a_word(merge_info, word):
	# merge_info: OrderedDict {('s','</w>'):0, ('e', '</w>'):1 ... }
	# word: "c e m e n t </w>" => "ce m e n t<\w>" 되어야 함.
	
	for info in merge_info:
		bigram = re.escape(' '.join(info))
		p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')

		# 만약 ''.join(info): r</w> 이고, word: 'a r </w>' 이면 w_out은 'a r</w>'가 된다.
		word = p.sub(''.join(info), word)
	return word



def make_bpe_format_testcode():
	path = "./testdata.en"

	word2idx = get_word_frequency_dict_for_bpe_from_document(path)	
	bpe2idx, merge_info, recover_info = get_bpe_information(word2idx)

	print('word2idx', len(word2idx))
	print('bpe2idx', len(bpe2idx))
	print('merge_info', len(merge_info))
	print('recover_info', len(recover_info), '\n\n')
		
	merge_a_word(recover_info, "c e m e n t </w>")
	for i in recover_info:
		print(i)	


def testcode_2():
	path = "./testdata.en"
	
	# 일단 다른 문서들로부터 bpe2idx, merge_info, recover_info를 구했다고 치자.
	word2idx = get_word_frequency_dict_for_bpe_from_document(path)	
	bpe2idx, merge_info, recover_info = get_bpe_information(word2idx, num_merges=30)#100

	#여기서부터 시작한다.
	bpe = document_preprocess_for_bpe(path, merge_info=merge_info) # 2d list not numpy
	# 쉽게 패딩하기. (pandas로 변환 => fillna 함수로 패딩처리 => numpy로 변환)
	bpe = pd.DataFrame(bpe)
	bpe.fillna('</p>', inplace=True) # NaN => pad_symbol	
	bpe = np.array(bpe)
	print(bpe)


	
	'''
	print(word2idx[0])	
	merge = merge_a_word(merge_info, word2idx[0][0])
	#merge = merge_a_word(recover_info, word2idx[0][0])
	print(merge)
	'''


#bpe_testcode()
#make_bpe_format_testcode()

testcode_2()
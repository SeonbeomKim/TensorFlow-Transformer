import argparse
import numpy as np
import csv
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument(
		'-mode', 
		help="train or infer",
		choices=['train', 'infer'],
		required=True, 
	)
parser.add_argument(
		'-source_input_path', 
		help="source document path",
		required=True, 
	)
parser.add_argument(
		'-source_out_path', 
		help="preprocessed source output path",
		required=True, 
	)
parser.add_argument(
		'-target_input_path', 
		help="target document path",
		required=True, 
	)
parser.add_argument(
		'-target_out_path', 
		help="preprocessed target output path",
		required=False
	)
parser.add_argument(
		'-bucket_out_path', 
		help="bucket output path",
		required=True, 
	)
parser.add_argument(
		'-voca_path', 
		help="Vocabulary_path",
		required=True
	)

args = parser.parse_args()

mode = args.mode 
source_input_path = args.source_input_path
source_out_path = args.source_out_path
target_input_path = args.target_input_path
target_out_path = args.target_out_path
bucket_out_path = args.bucket_out_path
voca_path = args.voca_path



def read_voca(path):
	sorted_voca = []
	with open(path, 'r', encoding='utf-8') as f:	
		for bpe_voca in f:
			bpe_voca = bpe_voca.strip()
			if bpe_voca:
				bpe_voca = bpe_voca.split()
				sorted_voca.append(bpe_voca)
	return sorted_voca



def make_bpe2idx(voca):
	bpe2idx = {'</p>':0, '</UNK>':1, '</g>':2, '</e>':3}	
	idx = 4

	for word, _ in voca:
		bpe2idx[word] = idx
		idx += 1

	return bpe2idx



def bpe2idx_out_csv(data_path, out_path, bpe2idx, info='source'): #info: 'source' or 'target'
	print('documents to idx csv', data_path, '->', out_path)

	o = open(out_path, 'w', newline='', encoding='utf-8')
	wr = csv.writer(o)

	with open(data_path, 'r', encoding='utf-8') as f:
		documents = f.readlines()

	for i in tqdm(range(len(documents)), ncols=50):
		sentence = documents[i]

		# bpe2idx
		if info == 'target':
			row_idx = [bpe2idx['</g>']]
		else:
			row_idx = []

		for word in sentence.strip().split():
			if word in bpe2idx:
				row_idx.append(bpe2idx[word])
			else:
				row_idx.append(bpe2idx['</UNK>']) ## 1
		row_idx.append(bpe2idx['</e>']) ## eos:3

		wr.writerow(row_idx)

	o.close()
	print('saved', out_path, '\n')



def _make_bucket_dataset(source_path, target_path, out_path, bucket, pad_idx, file_mode='w', is_trainset=True):
	print('make bucket dataset')
	print('source:', source_path, 'target:', target_path)

	if not os.path.exists(out_path):
		os.makedirs(out_path)

	# 저장시킬 object 생성 
	source_open_list = []
	target_open_list = []
	for bucket_size in bucket:
		o_s = open(os.path.join(out_path, 'source_'+str(bucket_size)+'.csv'), file_mode, newline='')
		o_s_csv = csv.writer(o_s)
		source_open_list.append((o_s, o_s_csv))
		
		if is_trainset:
			o_t = open(os.path.join(out_path, 'target_'+str(bucket_size)+'.csv'), file_mode, newline='')
			o_t_csv = csv.writer(o_t)
			target_open_list.append((o_t, o_t_csv))
		else:
			o_t = open(os.path.join(out_path, 'target_'+str(bucket_size)+'.txt'), file_mode, encoding='utf-8')
			target_open_list.append(o_t)



	with open(source_path, 'r') as s:
		source = s.readlines()

	if is_trainset:
		with open(target_path, 'r') as t:
			target = t.readlines()
	else:
		with open(target_path, 'r', encoding='utf-8') as t:
			target = t.readlines()


	for i in tqdm(range(len(source)), ncols=50):
		source_sentence = np.array(source[i].strip().split(','), dtype=np.int32)
		if is_trainset:
			target_sentence = np.array(target[i].strip().split(','), dtype=np.int32)
		else:
			target_sentence = target[i]


		for bucket_index, bucket_size in enumerate(bucket):
			source_size, target_size = bucket_size
			# 버켓에 없는것은 데이터는 제외.

			if is_trainset:
				if len(source_sentence) <= source_size and len(target_sentence) <= target_size: # (1,2) <= (10, 40)
					source_sentence = np.pad(
							source_sentence, 
							(0, source_size-len(source_sentence)),
							'constant',
							constant_values = pad_idx# bpe2idx['</p>'] # pad value
						)
					target_sentence = np.pad(
							target_sentence, 
							(0, target_size+1-len(target_sentence)), # [0:-1]: decoder_input, [1:]: decoder_target 이므로 +1 해줌.
							'constant',
							constant_values = pad_idx # bpe2idx['</p>'] # pad value
						)
					source_open_list[bucket_index][1].writerow(source_sentence)
					target_open_list[bucket_index][1].writerow(target_sentence)
					break

			else:
				if len(source_sentence) <= source_size:
					source_sentence = np.pad(
							source_sentence, 
							(0, source_size-len(source_sentence)),
							'constant',
							constant_values = pad_idx# bpe2idx['</p>'] # pad value
						)
					source_open_list[bucket_index][1].writerow(source_sentence)
					target_open_list[bucket_index].write(target_sentence)
					break				

	
	# close object 
	for i in range(len(bucket)):
		source_open_list[i][0].close()
		
		if is_trainset:
			target_open_list[i][0].close()
		else:
			target_open_list[i].close()
	print('saved', out_path)



def make_bucket_dataset(data_path, idx_out_path, bucket_out_path, bucket, bpe2idx, file_mode='w', is_trainset=True):
	print('start make_bucket_dataset', 'is_trainset:', is_trainset)

	bpe2idx_out_csv(
			data_path=data_path['source'], 
			out_path=idx_out_path['source'], 
			bpe2idx=bpe2idx, 
			info='source'
		)

	if is_trainset:
		bpe2idx_out_csv(
				data_path=data_path['target'], 
				out_path=idx_out_path['target'], 
				bpe2idx=bpe2idx, 
				info='target'
			) 

		# padding and bucketing
		_make_bucket_dataset(
				source_path=idx_out_path['source'], 
				target_path=idx_out_path['target'], 
				out_path=bucket_out_path, 
				bucket=bucket, 
				pad_idx=bpe2idx['</p>'],
				file_mode=file_mode,
				is_trainset=is_trainset
			)

	else:
		# padding and bucketing
		_make_bucket_dataset(
				source_path=idx_out_path['source'], 
				target_path=data_path['target'], 
				out_path=bucket_out_path, 
				bucket=bucket, 
				pad_idx=bpe2idx['</p>'],
				file_mode=file_mode,
				is_trainset=is_trainset
			)

	print('\n\n')



voca = read_voca(voca_path)
bpe2idx = make_bpe2idx(voca)




if mode == 'train':
	data_path = {'source':source_input_path, 'target':target_input_path}
	idx_out_path = {'source':source_out_path, 'target':target_out_path}
	
	#bucket  (source, target)
	train_bucket = [(i*5, i*5 + j*10) for i in range(1, 31) for j in range(4)]# [(5, 5), (5, 15), .., (5, 35), ... , (150, 150), .., (150, 180)]
	print('train_bucket\n', train_bucket,'\n')
	
	make_bucket_dataset(
			data_path, 
			idx_out_path, 
			bucket_out_path, 
			train_bucket, 
			bpe2idx
		)

elif mode == 'infer':
	data_path = {'source':source_input_path, 'target':target_input_path}
	idx_out_path = {'source':source_out_path}

	#bucket  (source, target)
	infer_bucket = [(i*5, i*5+50) for i in range(1, 31)] # [(5, 55), (10, 60), ..., (150, 200)]
	print('infer_bucket\n', infer_bucket,'\n')
	
	make_bucket_dataset(
			data_path, 
			idx_out_path, 
			bucket_out_path, 
			infer_bucket, 
			bpe2idx, 
			is_trainset=False
		)



'''
# make trainset
data_path = {'source':'./bpe_dataset/bpe_wmt17.en', 'target':'./bpe_dataset/bpe_wmt17.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_wmt17_en.csv', 'target':'./bpe_dataset/target_idx_wmt17_de.csv'}
bucket_out_path = './bpe_dataset/train_set_wmt17/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, train_bucket, bpe2idx)

# make validset
data_path = {'source':'./bpe_dataset/bpe_newstest2014.en', 'target':'./dataset/dev.tar/newstest2014.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2014_en.csv'}
bucket_out_path = './bpe_dataset/valid_set_newstest2014/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, is_trainset=False)

# make testset
data_path = {'source':'./bpe_dataset/bpe_newstest2015.en', 'target':'./dataset/dev.tar/newstest2015.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2015_en.csv'}
bucket_out_path = './bpe_dataset/test_set_newstest2015/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, is_trainset=False)

# make testset
data_path = {'source':'./bpe_dataset/bpe_newstest2016.en', 'target':'./dataset/dev.tar/newstest2016.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2016_en.csv'}
bucket_out_path = './bpe_dataset/test_set_newstest2016/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, is_trainset=False)
'''
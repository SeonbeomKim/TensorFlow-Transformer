import numpy as np
import csv
import os


def load_data(path, mode=None):
	data = np.load(path, encoding='bytes')
	if mode == 'dictionary':
		data = data.item()
	return data


def bpe2idx_out_csv(data_path, out_path, bpe2idx, read_line=None, info='source'): #info: 'source' or 'target'
	o = open(out_path, 'w', newline='', encoding='utf-8')
	wr = csv.writer(o)

	with open(data_path, 'r', encoding='utf-8') as f:
		for i, sentence in enumerate(f):
			if i == read_line:
				break

			if (i+1) % 500000 == 0:
				print(out_path, i+1)

			# bpe2idx
			if info == 'target':
				row_idx = [bpe2idx['</g>']]
			else:
				row_idx = []

			for word in sentence.split():
				if word in bpe2idx:
					row_idx.append(bpe2idx[word])
				else:
					row_idx.append(bpe2idx['UNK']) ## 0
			
			if info == 'target':
				row_idx.append(bpe2idx['</e>']) ## eos:2

			wr.writerow(row_idx)

	o.close()
	print('saved', out_path, '\n')



def source_target_bucketing_and_concat_out_csv(source_path, target_path, out_path, bucket, pad_idx, file_mode='w'):
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	# 저장시킬 object 생성 
	open_list = []
	for bucket_size in bucket:
		o = open(out_path+str(bucket_size)+'.csv', file_mode, newline='')
		o_csv = csv.writer(o)
		open_list.append((o, o_csv))


	with open(source_path, 'r', newline='') as source, open(target_path, 'r', newline='') as target:
		source_wr = csv.reader(source)
		target_wr = csv.reader(target)

		for i, sentence in enumerate(zip(source_wr, target_wr)):
			if (i+1) % 500000 == 0:
				print('line:', i+1)

			source_sentence = np.array(sentence[0], dtype=np.int32)
			target_sentence = np.array(sentence[1], dtype=np.int32)
			
			for bucket_index, bucket_size in enumerate(bucket):
				source_size, target_size = bucket_size
				if len(source_sentence) <= source_size and len(target_sentence) <= target_size: # (1,2) <= (10, 40)
					source_sentence = np.pad(
							source_sentence, 
							(0, source_size-len(source_sentence)),
							'constant',
							constant_values = pad_idx# bpe2idx['</p>'] # pad value
						)
					target_sentence = np.pad(
							target_sentence, 
							(0, target_size-len(target_sentence)),
							'constant',
							constant_values = pad_idx # bpe2idx['</p>'] # pad value
						)
					open_list[bucket_index][1].writerow(np.concatenate((source_sentence, target_sentence)))	
					break
	
	# close object 
	for o, _ in open_list:
		o.close()
	print('saved', out_path, '\n')



# source는 idx->bucketing, target은 원본 그대로. 둘이 concat
def source_bucketing_and_concat_out_csv(source_path, target_path, out_path, bucket, pad_idx, file_mode='w'):
	if not os.path.exists(out_path):
		os.makedirs(out_path)

	# 저장시킬 object 생성 
	open_list = []
	for bucket_size in bucket:
		o = open(out_path+str(bucket_size)+'.csv', file_mode, newline='', encoding='utf-8')
		o_csv = csv.writer(o)
		open_list.append((o, o_csv))


	with open(source_path, 'r', newline='') as source, open(target_path, 'r', newline='', encoding='utf-8') as target:
		source_wr = csv.reader(source)
		target_wr = csv.reader(target)

		for i, sentence in enumerate(zip(source_wr, target_wr)):
			if (i+1) % 500000 == 0:
				print('line:', i+1)

			source_sentence = np.array(sentence[0], dtype=np.int32)
			target_sentence = sentence[1][0].split()

			for bucket_index, bucket_size in enumerate(bucket):
				source_size, target_size = bucket_size
				if len(source_sentence) <= source_size:
					source_sentence = np.pad(
							source_sentence, 
							(0, source_size-len(source_sentence)),
							'constant',
							constant_values = pad_idx# bpe2idx['</p>'] # pad value
						)
					open_list[bucket_index][1].writerow(np.concatenate((source_sentence, target_sentence)))	
					break
	
	# close object 
	for o, _ in open_list:
		o.close()
	print('saved', out_path, '\n')




def make_train_dataset_out_csv(source_target_path, source_target_idx_out_path, dataset_out_path, bucket, bpe2idx, read_line=None):
	print('start make_dataset_with_target')
	print('source:', source_target_path[0], 'idx_out_source:', source_target_idx_out_path[0])
	print('target:', source_target_path[1], 'idx_out_target:', source_target_idx_out_path[1])
	print('dataset:', dataset_out_path, '\n')

	bpe2idx_out_csv(
			data_path=source_target_path[0], 
			out_path=source_target_idx_out_path[0], 
			bpe2idx=bpe2idx, 
			read_line=read_line, 
			info='source'
		)

	bpe2idx_out_csv(
			data_path=source_target_path[1], 
			out_path=source_target_idx_out_path[1], 
			bpe2idx=bpe2idx, 
			read_line=read_line, 
			info='target'
		) 

	# idx 데이터들 버켓팅(패딩포함)하고, concat
	source_target_bucketing_and_concat_out_csv(
			source_path=source_target_idx_out_path[0], 
			target_path=source_target_idx_out_path[1], 
			out_path=dataset_out_path, 
			bucket=bucket, 
			pad_idx=bpe2idx['</p>']
		)


def make_valid_test_dataset_out_csv(source_target_path, source_idx_out_path, dataset_out_path, bucket, bpe2idx, read_line=None, file_mode='w'):
	print('start make_dataset_without_target')
	print('source:', source_target_path[0], 'idx_out_source:', source_idx_out_path)
	print('target:', source_target_path[1])
	print('dataset:', dataset_out_path, '\n')

	bpe2idx_out_csv(
			data_path=source_target_path[0], 
			out_path=source_idx_out_path, 
			bpe2idx=bpe2idx, 
			read_line=read_line, 
			info='source'
		)

	source_bucketing_and_concat_out_csv(
			source_path=source_idx_out_path, 
			target_path=source_target_path[1], 
			out_path=dataset_out_path, 
			bucket=bucket, 
			pad_idx=bpe2idx['</p>'], 
			file_mode=file_mode
		)

# (source, target)
bucket = [(10, 40), (30, 60), (50, 80), (70, 100), (100, 130), (140, 170), (180, 210)]
bpe2idx_path = './npy/bpe2idx.npy'
bpe2idx = load_data(bpe2idx_path, mode='dictionary')

# make trainset
source_target_path = ['./bpe_dataset/bpe_wmt17.en', './bpe_dataset/bpe_wmt17.de']
source_target_out_path = ['./bpe_dataset/source_idx_wmt17_en.csv', './bpe_dataset/target_idx_wmt17_de.csv']
dataset_out_path = './bpe_dataset/train_set/'
make_train_dataset_out_csv(source_target_path, source_target_out_path, dataset_out_path, bucket, bpe2idx, read_line=None)

# make validset
source_target_path = ['./bpe_dataset/bpe_newstest2014.en', './dataset/dev.tar/newstest2014.tc.de']
source_idx_out_path = './bpe_dataset/source_idx_newstest2014_en.csv'
dataset_out_path = './bpe_dataset/valid_set/'
make_valid_test_dataset_out_csv(source_target_path, source_idx_out_path, dataset_out_path, bucket, bpe2idx, read_line=None)

# make testset
source_target_path = ['./bpe_dataset/bpe_newstest2015.en', './dataset/dev.tar/newstest2015.tc.de']
source_idx_out_path = './bpe_dataset/source_idx_newstest2015_en.csv'
dataset_out_path = './bpe_dataset/test_set/'
make_valid_test_dataset_out_csv(source_target_path, source_idx_out_path, dataset_out_path, bucket, bpe2idx, read_line=None)

# make testset
source_target_path = ['./bpe_dataset/bpe_newstest2016.en', './dataset/dev.tar/newstest2016.tc.de']
source_idx_out_path = './bpe_dataset/source_idx_newstest2016_en.csv'
dataset_out_path = './bpe_dataset/test_set/'
make_valid_test_dataset_out_csv(source_target_path, source_idx_out_path, dataset_out_path, bucket, bpe2idx, read_line=None, file_mode='a')


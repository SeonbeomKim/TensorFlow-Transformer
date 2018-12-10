import numpy as np
import csv
import os


############확인용 코드########################
def get_maximum_length(data_path_list, read_line=500000):
	maximum = 0
	for path in data_path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for i, sentence in enumerate(f):
				if i == read_line:
					break

				row_length = len(sentence.split())
				if row_length > maximum:
					maximum = row_length

	return maximum  


def get_all_length_for_bucketing_boundary(data_path_list, read_line=500000):

	length_dict = {}

	for path in data_path_list:
		with open(path, 'r', encoding='utf-8') as f:
			for i, sentence in enumerate(f):
				if i == read_line:
					break

				row_length = len(sentence.split())
				if row_length not in length_dict:
					length_dict[row_length] = 1
				else:
					length_dict[row_length] += 1
	
	sorted_list = []
	for key in length_dict:
		sorted_list.append([key, length_dict[key]])

	return sorted(sorted_list)
	#return length_dict
#############################################



def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data

def save_dictionary(path, dictionary):
	np.save(path, dictionary)



def bpe_to_csv(data_path, out_name, bpe2idx, read_line=500000, info='source', write_mode='w'): #info: 'source' or 'target'
	cache = load_dictionary(npy_path+'cache.npy')

	o = open(out_name, write_mode, newline='', encoding='utf-8')
	wr = csv.writer(o)

	with open(data_path, 'r', encoding='utf-8') as f:
		for i, sentence in enumerate(f):
			if i == read_line:
				break

			if (i+1) % 10000 == 0:
				print(out_name, i+1, '/', read_line)

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
	print('saved', out_name)



def source_target_bucketing_and_concat(input_name, target_name, bucket, bpe2idx):

	bucket_dict = {}
	for bucket_list in bucket:
		bucket_dict[bucket_list] = [] # key 초기화.

	
	with open(input_name, 'r', newline='') as source, open(target_name, 'r', newline='') as target:
		source_wr = csv.reader(source)
		target_wr = csv.reader(target)

		for line, data in enumerate(zip(source_wr, target_wr)):

			if (line+1) % 10000 == 0:
				print('line:', line+1)

			source_sentence = np.array(data[0], dtype=np.int32)
			target_sentence = np.array(data[1], dtype=np.int32)
			
			for bucket_list in bucket:
				if len(source_sentence) <= bucket_list[0] and len(target_sentence) <= bucket_list[1]: # (1,2) <= (10, 30)		

					source_sentence = np.pad(
								source_sentence, 
								(0, bucket_list[0]-len(source_sentence)),
								'constant',
								constant_values = bpe2idx['</p>'] # pad value
							)
					target_sentence = np.pad(
								target_sentence, 
								(0, bucket_list[1]-len(target_sentence)),
								'constant',
								constant_values = bpe2idx['</p>'] # pad value
							)
						
					bucket_dict[bucket_list].append(np.concatenate((source_sentence, target_sentence)))
					break
					
					
	for key in bucket_dict:
		bucket_dict[key] = np.array(bucket_dict[key])

	save_dictionary(npy_path + 'bucket_concat_dataset.npy', bucket_dict)
	return bucket_dict



def split_train_validation(bucket_concat_dict, vali_ratio=0.1):
	
	train_bucket_dict = {}
	validation_bucket_dict = {}

	for key in bucket_concat_dict:
		if len(bucket_concat_dict[key]) > 0:
			np.random.shuffle(bucket_concat_dict[key])

			vali_length = int(np.ceil(len(bucket_concat_dict[key])*vali_ratio))
			validation = bucket_concat_dict[key][:vali_length]
			train = bucket_concat_dict[key][vali_length:]
			print('key:', key)
			print('# train', len(train))
			print('# validation', len(validation), '\n')

			train_bucket_dict[key] = train
			validation_bucket_dict[key] = validation

	save_dictionary(npy_path + 'train_bucket_concat_dataset.npy', train_bucket_dict)
	save_dictionary(npy_path + 'valid_bucket_concat_dataset.npy', validation_bucket_dict)

	print('saved', npy_path+'train_bucket_concat_dataset.npy', npy_path+'valid_bucket_concat_dataset.npy')



def bpe2idx_make_train_valid_set(bpe_data_list, out_name_list, bucket, vali_ratio=0.1):
	#maximum = get_maximum_length(data) #143임
	
	#bpe2idx
	bpe2idx = load_dictionary(npy_path+'bpe2idx.npy')

	# bpe2idx csv 생성.
	bpe_to_csv(bpe_data_list[0], out_name_list[0], bpe2idx, read_line=500000, info='source')
	bpe_to_csv(bpe_data_list[1], out_name_list[1], bpe2idx, read_line=500000, info='target')

	#source_target bucketing and concat
	bucket_concat_dict = source_target_bucketing_and_concat(out_name_list[0], out_name_list[1], bucket, bpe2idx)

	#split and save as npy format
	split_train_validation(bucket_concat_dict, vali_ratio)


def bpe2idx_make_test_set(bpe_data_list, out_name_list):
	bpe2idx = load_dictionary(npy_path+'bpe2idx.npy')

	for data_path in bpe_data_list:
		bpe_to_csv(data_path, out_name_list[0], bpe2idx, info='source', write_mode='a') # 한 파일에 테스트셋 합치기.


# (source, target)
bucket = [(10, 30), (20, 40), (50, 70), (80, 100), (110, 140), (150, 170), (180, 200)]

npy_path = './bpe_dataset/'
'''
bpe_data_list = ['./bpe_dataset/bpe_wmt17.en', './bpe_dataset/bpe_wmt17.de']
out_name = ['./bpe_dataset/bpe2idx_en.csv', './bpe_dataset/bpe2idx_de.csv']
bpe2idx_make_train_valid_set(bpe_data_list, out_name, bucket, vali_ratio=0.1)
'''

bpe_test_data_list = ['./bpe_dataset/bpe_newstest2014.en', './bpe_dataset/bpe_newstest2015.en', './bpe_dataset/bpe_newstest2016.en']
test_out_name = ['./bpe_dataset/testset.csv']
bpe2idx_make_test_set(bpe_test_data_list, test_out_name)


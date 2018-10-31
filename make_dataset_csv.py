import numpy as np
import csv
import re, collections
import os

save_path = './wmt17_dataset/'
in_path = './wmt17_dataset/'

def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data


def bpe_to_csv(data_path, out_name, bpe2idx, read_line=500000, maximum_length=200):
	cache = load_dictionary(in_path+'cache.npy')

	o = open(save_path + out_name, 'w', newline='', encoding='utf-8')
	wr = csv.writer(o)

	with open(in_path + data_path, 'r', encoding='utf-8') as f:
		for i, sentence in enumerate(f):
			if i == read_line:
				break

			if (i+1) % 10000 == 0:
				print(out_name, i+1, '/', read_line)

			row_idx = []
			for word in sentence.split():
				if word in bpe2idx:
					row_idx.append(bpe2idx[word])
				else:
					row_idx.append(bpe2idx['UNK']) ## 0
			
			row_idx.append(bpe2idx['</e>']) ## eos:2
			sequence_length = len(row_idx)

			row_idx = np.pad(row_idx, (0, maximum_length-len(row_idx)), 'constant', constant_values = bpe2idx['</p>']) # pad:3
			row_idx = np.append(row_idx, sequence_length)
			wr.writerow(row_idx)

	o.close()
	print('saved', out_name)


def get_maximum_length(data_path_list, read_line=500000):
	maximum = 0
	for path in data_path_list:
		with open(in_path + path, 'r', encoding='utf-8') as f:
			for i, sentence in enumerate(f):
				if i == read_line:
					break

				row_length = len(sentence.split())
				if row_length > maximum:
					maximum = row_length

	return maximum  


def concat_input_target(input_name, target_name):
	temp_1 = []
	with open(save_path + input_name, 'r', newline='') as inp_f:
		wr = csv.reader(inp_f)
		for i in wr:
			temp_1.append(i)
		temp_1 = np.array(temp_1)

	temp_2 = []
	with open(save_path + target_name, 'r', newline='') as tar_f:
		wr = csv.reader(tar_f)
		for i in wr:
			temp_2.append(i)
		temp_2 = np.array(temp_2)
	concat = np.hstack((temp_1, temp_2))
	print('concat complete', 'concat shape', concat.shape)

	return concat


def split_train_validation(concat, vali_ratio=0.1):
	np.random.shuffle(concat)
	vali_length = int(np.ceil(len(concat)*vali_ratio))
	
	validation = concat[:vali_length]
	train = concat[vali_length:]
	print('# train', len(train))
	print('# validation', len(validation))

	with open(save_path+'train.csv', 'w', newline='') as o:
		wr = csv.writer(o)
		for i in train:
			wr.writerow(i)

	with open(save_path+'validation.csv', 'w', newline='') as o:
		wr = csv.writer(o)
		for i in validation:
			wr.writerow(i)

	print('saved', save_path+'train.csv', save_path+'validation.csv')


def bpe2idx_and_split_dataset(bpe_data_list, out_name_list, vali_ratio=0.1):
	#maximum = get_maximum_length(data) #143임. 약 200개 잡고 하자.
	
	if not os.path.exists(save_path):
		print("create save directory")
		os.makedirs(save_path)

	#bpe2idx
	bpe2idx = load_dictionary(in_path+'bpe2idx.npy')

	bpe_to_csv(bpe_data_list[0], out_name_list[0], bpe2idx, read_line=500000, maximum_length=200)
	bpe_to_csv(bpe_data_list[1], out_name_list[1], bpe2idx, read_line=500000, maximum_length=200)

	#train_target concat
	concat = concat_input_target(out_name_list[0], out_name_list[1])

	#split and save as csv format
	split_train_validation(concat, vali_ratio)



data = ['bpe_wmt17.en', 'bpe_wmt17.de']
out_name = ['bpe2idx_en.csv', 'bpe2idx_de.csv']
bpe2idx_and_split_dataset(data, out_name, vali_ratio=0.1)
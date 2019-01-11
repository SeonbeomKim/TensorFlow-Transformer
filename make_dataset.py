import numpy as np
import csv
import os
from tqdm import tqdm

def save_data(path, data):
	np.save(path, data)

def load_data(path, mode=None):
	data = np.load(path, encoding='bytes')
	if mode == 'dictionary':
		data = data.item()
	return data


def make_bpe2idx(voca, npy_path):
	bpe2idx = {'</p>':0, '</UNK>':1, '</g>':2, '</e>':3}	
	idx2bpe = ['</p>', '</UNK>', '</g>', '</e>']
	idx = 4

	for word, _ in voca:
		bpe2idx[word] = idx
		idx += 1
		idx2bpe.append(word)

	save_data(npy_path+'bpe2idx.npy', bpe2idx)
	save_data(npy_path+'idx2bpe.npy', idx2bpe)
	print('save bpe2idx, size:', len(bpe2idx))
	print('save idx2bpe, size:', len(idx2bpe))
	return bpe2idx, idx2bpe



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
		o_s = open(out_path+'source_'+str(bucket_size)+'.csv', file_mode, newline='')
		o_s_csv = csv.writer(o_s)
		source_open_list.append((o_s, o_s_csv))
		
		if is_trainset:
			o_t = open(out_path+'target_'+str(bucket_size)+'.csv', file_mode, newline='')
			o_t_csv = csv.writer(o_t)
			target_open_list.append((o_t, o_t_csv))
		else:
			o_t = open(out_path+'target_'+str(bucket_size)+'.txt', file_mode, encoding='utf-8')
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



#bucket  (source, target)
train_bucket = [(i*5, i*5 + j*10) for i in range(1, 31) for j in range(4)]# [(5, 5), (5, 15), .., (5, 35), ... , (150, 150), .., (150, 180)]
infer_bucket = [(i*5, i*5+50) for i in range(1, 31)] # [(5, 55), (10, 60), ..., (150, 200)]
print('train_bucket\n', train_bucket,'\n')
print('infer_bucket\n', infer_bucket,'\n')

npy_path = './npy/'
voca_path = npy_path+'final_voca.npy'
final_voca_threshold = 50

voca = load_data(voca_path)
print('original voca size:', len(voca))
voca = [(word, int(freq)) for (word, freq) in voca if int(freq) >= final_voca_threshold]
print('threshold applied voca size:', len(voca), '\n')
bpe2idx, _ = make_bpe2idx(voca, npy_path)


# make trainset
data_path = {'source':'./bpe_dataset/bpe_wmt17.en', 'target':'./bpe_dataset/bpe_wmt17.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_wmt17_en.csv', 'target':'./bpe_dataset/target_idx_wmt17_de.csv'}
bucket_out_path = './bpe_dataset/train_set/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, train_bucket, bpe2idx)

# make validset
data_path = {'source':'./bpe_dataset/bpe_newstest2014.en', 'target':'./dataset/dev.tar/newstest2014.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2014_en.csv'}
bucket_out_path = './bpe_dataset/valid_set/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, is_trainset=False)

# make testset
data_path = {'source':'./bpe_dataset/bpe_newstest2015.en', 'target':'./dataset/dev.tar/newstest2015.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2015_en.csv'}
bucket_out_path = './bpe_dataset/test_set/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, is_trainset=False)

# make testset
data_path = {'source':'./bpe_dataset/bpe_newstest2016.en', 'target':'./dataset/dev.tar/newstest2016.tc.de'}
idx_out_path = {'source':'./bpe_dataset/source_idx_newstest2016_en.csv'}
bucket_out_path = './bpe_dataset/test_set/'
make_bucket_dataset(data_path, idx_out_path, bucket_out_path, infer_bucket, bpe2idx, file_mode='a', is_trainset=False)

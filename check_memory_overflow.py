import csv
import numpy as np
import bucket_data_helper

def _read_csv(path):
	print('read csv data', path)
	data = np.loadtxt(
			path, 
			delimiter=",", 
			dtype=np.int32,
			ndmin=2 # csv가 1줄이여도 2차원으로 출력.
		)

	return data

def _read_txt(path):
	print('read txt data', path)
	data = []
	with open(path, 'r', encoding='utf-8') as f:
		for sentence in f:
			# EOF check
			if sentence == '\n' or sentence == ' ' or sentence == '':
				break
			if sentence[-1] == '\n':
				sentence = sentence[:-1]
			data.append(sentence.split())					
		return data

def read_data_set(sentence_path, target_path, bucket, target_type='csv'):
	dictionary = {}
	for bucket_size in bucket:
		sentence = _read_csv(sentence_path+str(bucket_size)+'.csv')
		
		if target_type == 'csv':
			target = _read_csv(target_path+str(bucket_size)+'.csv')
		else:
			target = _read_txt(target_path+str(bucket_size)+'.txt')

		# 개수가 0인 bucket은 버림.
		if len(sentence) != 0:
			dictionary[bucket_size] = [sentence, target]
		
			if target_type =='csv':
				print(sentence.shape, target.shape, '\n')
			else:
				print(sentence.shape, len(target), '\n')

	print('\n\n')
	return dictionary
		
bucket = [(10, 40), (30, 60), (50, 80), (70, 100), (100, 130), (140, 170), (180, 210)]
train_sentence_path = './bpe_dataset/train_set/source_'
train_target_path = './bpe_dataset/train_set/target_'

valid_sentence_path = './bpe_dataset/valid_set/source_'
valid_target_path = './bpe_dataset/valid_set/target_'

test_sentence_path = './bpe_dataset/test_set/source_'
test_target_path = './bpe_dataset/test_set/target_'

#train_dict = read_data_set(train_sentence_path, train_target_path, bucket)
#valid_dict = read_data_set(valid_sentence_path, valid_target_path, bucket, 'txt')
test_dict = read_data_set(test_sentence_path, test_target_path, bucket, 'txt')

#train_set = bucket_data_helper.bucket_data(train_dict, iter=True, batch_token = 12000) # batch_token // len(sentence||target token) == batch_size
#valid_set = bucket_data_helper.bucket_data(valid_dict, iter=True, batch_token = 12000) # batch_token // len(sentence||target token) == batch_size
test_set = bucket_data_helper.bucket_data(test_dict, iter=True, batch_token = 15000) # batch_token // len(sentence||target token) == batch_size


#test_set.shuffle()
 

#### test 140 170 이거 데이터 1갠데 140개로 나옴. 체크 ㄱㄱ

count = {}

print('iter_fn', test_set.get_number_of_iter())
for k in range(2):
	for i in range(test_set.get_number_of_iter()):
		batch = test_set.get_batch()
		if batch[-1] in count:
			count[batch[-1]] += len(batch[0])
		else:
			count[batch[-1]] = len(batch[0])	

print('\n\n')
sum = 0
for i in count:
	print(i, count[i])
	sum += count[i]
print(sum)

print('iter', zz)
print('iter_fn2', test_set.get_number_of_iter())

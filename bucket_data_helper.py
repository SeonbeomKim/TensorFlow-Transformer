import numpy as np

# get_batch 함수로 데이터 긁어옴. return이 0 이면 epoch 끝. 
class bucket_data:  
	def __init__(self, data, iter=True, batch_token = 16000):
		self.data = data
		self.iter = iter
		self.key = list(data.keys())

		self.key_length = len(self.key)
		self.key_index = 0
		self.data_index = 0
		self.data_length = len(self.data[self.key[self.key_index]])

		self.batch_token = batch_token


	def get_batch(self):
		# data_index: 3, data_length: 3이면, data:[1,2,3] 가져올 데이터가 없으므로 키 증가.
		if self.data_index >= self.data_length:
			
			# 아직 추출할 bucket이 남았다면
			if self.key_index < self.key_length - 1:
				self.key_index += 1 # 다음 키로 넘어가고,
				self.data_index = 0 # 데이터 인덱스 0으로 초기화.
				self.data_length = len(self.data[self.key[self.key_index]]) # 다음 키에 관한 데이터 길이.

			# 모든 데이터 추출 완료.
			else:
				# iteration을 위해서 초기화.
				self.key_index = 0
				self.data_index = 0
				self.data_length = len(self.data[self.key[self.key_index]]) # 첫 키에 관한 데이터 길이.
				return 0


		bucket_data = self.data[self.key[self.key_index]]
		
		batch_size = self.batch_token // sum(self.key[self.key_index])
		batch = bucket_data[self.data_index:self.data_index + batch_size]		
		self.data_index += batch_size

		return batch, self.key[self.key_index] # ex) batch, (10, 30)


	def get_all_bucket(self):
		return self.key		


	def shuffle(self):
		for i in self.data:
			np.random.shuffle(self.data[i])

'''

import numpy as np
def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data


valid_set_path = './bpe_dataset/valid_bucket_concat_dataset.npy'
valid_set = load_dictionary(valid_set_path)

test = bucket_data(valid_set)

length_check2 = {}

while True:
	batch = test.get_batch()
	if batch is 0:
		break
	batch_data, bucket = batch[0], batch[1]
	#print(batch_data.shape, bucket)
	print(batch_data[0])
	break
'''



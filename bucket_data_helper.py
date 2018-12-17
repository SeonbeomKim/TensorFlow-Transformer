import numpy as np

# get_batch 함수로 데이터 긁어옴. return이 0 이면 epoch 끝. 
class bucket_data:  
	def __init__(self, data, iter=True, batch_token = 16000):
		self.data = data
		self.iter = iter
		self.key = list(data.keys()) #bucket

		self.key_length = len(self.key)
		self.key_index = 0
		self.data_index = 0
		self.data_length = len(self.data[self.key[self.key_index]][0]) #[0]:source, [1]:target
		self.batch_token = batch_token

		self.total_iter = self.get_number_of_iter()
		self.current_iter = 0


	def get_batch(self):
		if self.current_iter == self.total_iter:
			self.reset_for_iter()
			
		# data_index: 3, data_length: 3이면, data:[1,2,3] 가져올 데이터가 없으므로 키 증가.
		elif (self.data_index >= self.data_length) and (self.key_index < self.key_length - 1): # 아직 추출할 bucket이 남았다면
			self.set_next_key_for_iter()

		bucket_data = self.data[self.key[self.key_index]]
		batch_size = self.batch_token // sum(self.key[self.key_index]) 
		batch_source = bucket_data[0][self.data_index:self.data_index + batch_size]		
		batch_target = bucket_data[1][self.data_index:self.data_index + batch_size]		
		self.data_index += batch_size
		self.current_iter += 1

		return batch_source, batch_target, self.key[self.key_index] # ex) batch, (10, 30)


	def get_number_of_iter(self):
		num_iter = 0
		for key in self.data:
			batch_size = self.batch_token // sum(key) # batch_token //sum(@@) 이 0인것을 대비함. 
			num_iter += int(np.ceil(len(self.data[key][0])/batch_size))
		return num_iter


	def set_next_key_for_iter(self):
		self.key_index += 1 # 다음 키로 넘어가고,
		self.data_index = 0 # 데이터 인덱스 0으로 초기화.
		self.data_length = len(self.data[self.key[self.key_index]][0]) # 다음 키에 관한 데이터 길이.


	def reset_for_iter(self):
	# iteration을 위해서 초기화.
		self.key_index = 0
		self.data_index = 0
		self.current_iter = 0
		np.random.shuffle(self.key) # 키 순서 섞음.
		self.data_length = len(self.data[self.key[0]][0]) # 첫 키에 관한 데이터 길이.	


	def get_all_bucket(self):
		return self.key		


	def shuffle(self):
		for i in self.data:
			source, target = self.data[i]
			indices = np.arange(len(source))
			np.random.shuffle(indices)
			self.data[i] = [source[indices], target[indices]]

	'''
	def get_batch(self):
		print(self.key[self.key_index], self.data_index, self.data_length)
		#print(self.data_length)
		# data_index: 3, data_length: 3이면, data:[1,2,3] 가져올 데이터가 없으므로 키 증가.
		if self.data_index >= self.data_length:
			
			# 아직 추출할 bucket이 남았다면
			if self.key_index < self.key_length - 1:
				self.set_next_key_for_iter()

			# 모든 데이터 추출 완료.
			else:
				self.reset_for_iter()
				return 0

		bucket_data = self.data[self.key[self.key_index]]
		batch_size = max(self.batch_token // sum(self.key[self.key_index]), sum(self.key[self.key_index])) # batch_token //sum(@@) 이 0인것을 대비함. 
		batch_source = bucket_data[0][self.data_index:self.data_index + batch_size]		
		batch_target = bucket_data[1][self.data_index:self.data_index + batch_size]		
		self.data_index += batch_size

		return batch_source, batch_target, self.key[self.key_index] # ex) batch, (10, 30)
	'''

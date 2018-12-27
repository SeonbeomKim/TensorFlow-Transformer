import numpy as np

class bucket_data:  
	def __init__(self, data, batch_token = 16000):
		self.data = data
		self.batch_token = batch_token


	def get_dataset(self, bucket_shuffle=False, dataset_shuffle=False):
		# bucket_shuffle: 버켓별로 셔플.
		# dataset_shuffle: data_list 셔플 
		if bucket_shuffle is True:
			self.shuffle()

		data_list = []
		for key in self.data:
			batch_size = self.batch_token // sum(key) # batch_token //sum(@@) 이 0인것을 대비함. 
			
			for i in range( int(np.ceil(len(self.data[key][0])/batch_size)) ):
				bucket_data = self.data[key]
				batch_source = bucket_data[0][i*batch_size : (i+1)*batch_size]		
				batch_target = bucket_data[1][i*batch_size : (i+1)*batch_size]		
				data_list.append([batch_source, batch_target])

		if dataset_shuffle is True:
			np.random.shuffle(data_list)
		return data_list


	def shuffle(self):
		for key in self.data:
			source, target = self.data[key]
			indices = np.arange(len(source))
			np.random.shuffle(indices)
			self.data[key] = [source[indices], target[indices]]


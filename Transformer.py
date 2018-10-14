#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization

import tensorflow as tf #version 1.4
import numpy as np
import os

#tf.set_random_seed(777)

class Transformer:
	def __init__(self, sess, sentence_length=20, target_length=20, voca_size=50000, embedding_size=512, go_idx=1, eos_idx=-2, lr=0.01):
		self.sentence_length = sentence_length #encoder
		self.target_length = target_length #decoder (include eos)
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.go_idx = go_idx # <'go'> symbol index
		self.eos_idx = eos_idx # <'eos'> symbol index
		self.lr = lr
		self.PE = self.positional_encoding() #[self.target_length + alpha, self.embedding_siz] #slice해서 쓰자.
		

		with tf.name_scope("placeholder"):	
			self.sentence = tf.placeholder(tf.int32, [None, self.sentence_length]) 
			self.sentence_sequence_length = tf.placeholder(tf.int32, [None]) # no padded length
			self.target = tf.placeholder(tf.int32, [None, self.target_length])
			self.target_sequence_length = tf.placeholder(tf.int32, [None])  # include eos
			#self.keep_prob = tf.placeholder(tf.float32)
			
		with tf.name_scope('masks'):
			# https://www.tensorflow.org/api_docs/python/tf/sequence_mask
			self.sentence_mask =  tf.sequence_mask(  # [N, self.sentence_length] 
					self.sentence_sequence_length, 
					maxlen=self.sentence_length, 
					dtype=tf.float32
				)
			self.target_mask = tf.sequence_mask( # [N, target_sequence_length] (include eos)
					self.target_sequence_length, 
					maxlen=self.target_length, 
					dtype=tf.float32
				) 
			self.decoder_mask = tf.sequence_mask( # [target_sequence_length, target_sequence_length]
					tf.constant([i for i in range(1, self.target_length+1)]),
					maxlen=self.target_length,
					dtype=tf.float32
				)

		with tf.name_scope("embedding_table"):
			self.input_embedding_table = tf.Variable(tf.random_normal([self.voca_size-2, self.embedding_size])) #-2(except eos, go symbol) 
			self.output_embedding_table = tf.Variable(tf.random_normal([self.voca_size, self.embedding_size])) 


		with tf.name_scope('encoder'):
			self.encoder_embedding = self.encoder() # [N, self.sentence_length, self.embedding_size]


		with tf.name_scope('decoder'):
			with tf.name_scope('train'):
				self.train_pred_embedding = self.train_helper(self.encoder_embedding) # [N, self.target_length, self.voca_size]
				self.train_pred = tf.argmax(self.train_pred_embedding, axis=-1, output_type=tf.int32) # [N, self,target_length]
				
				# find first eos index  ex [5, 6, 4, 5, 5]
				self.train_first_eos = tf.argmax( tf.cast( tf.equal(self.train_pred, self.eos_idx), tf.int32 ), axis=-1) # [N]
				self.train_eos_mask = tf.sequence_mask(
						self.train_first_eos,
						maxlen=self.target_length,
						dtype=tf.int32
					)
				self.train_pred_except_eos = self.train_pred * self.train_eos_mask
				self.train_pred_except_eos += (self.train_eos_mask - 1) # excepted position value is -1


			with tf.name_scope('inference'):
				self.infer_pred_embedding = self.infer_helper(self.encoder_embedding) # [N, self.target_length, self.voca_size]
				self.infer_pred = tf.argmax(self.infer_pred_embedding, axis=-1, output_type=tf.int32) # [N, self,target_length]
				
				# find first eos index  ex [5, 6, 4, 5, 5]
				self.infer_first_eos = tf.argmax( tf.cast( tf.equal(self.infer_pred, self.eos_idx), tf.int32 ), axis=-1) # [N]
				self.infer_eos_mask = tf.sequence_mask(
						self.infer_first_eos,
						maxlen=self.target_length,
						dtype=tf.int32
					)
				self.infer_pred_except_eos = self.infer_pred * self.infer_eos_mask
				self.infer_pred_except_eos += (self.infer_eos_mask - 1) # excepted position value is -1
			

		with tf.name_scope('cost'): 
			# https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss #내부에서 weighted softmax cross entropy 동작.
			self.train_cost = tf.contrib.seq2seq.sequence_loss(self.train_pred_embedding, self.target, self.target_mask)
			self.infer_cost = tf.contrib.seq2seq.sequence_loss(self.infer_pred_embedding, self.target, self.target_mask) 
	

		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9) 
			self.minimize = optimizer.minimize(self.train_cost)


		with tf.name_scope('correct_check'):
			# target except eos
			target_eos_mask =  tf.sequence_mask( 
					self.target_sequence_length - 1,  #except eos
					maxlen=self.target_length, 
					dtype=tf.int32
				) 
			target_except_eos = self.target * target_eos_mask # except eos and pad
			target_except_eos += (target_eos_mask - 1) #  eos and pad position value is -1
			
			# correct check
			check_equal_position = tf.cast(tf.equal(target_except_eos, self.infer_pred_except_eos), dtype=tf.float32) # [N, self.target_length]
			#if use mean, 0.9999999 is equal to 1, so use sum.
			check_equal_position_sum = tf.reduce_sum(check_equal_position, axis=-1) # [N]
			#if correct: "check_equal_position_sum" value is equal to self.target_length
			correct_check = tf.cast(tf.equal(check_equal_position_sum, self.target_length), tf.float32) # [N]
			self.correct_count = tf.reduce_sum(correct_check) # scalar


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)
		
		sess.run(tf.global_variables_initializer())


	def encoder(self):
		#[N, self.sentence_length, self.embedding_size]
		encoder_input = tf.nn.embedding_lookup(self.input_embedding_table, self.sentence) 
		encoder_input += self.PE[:self.sentence_length, :] 
		mask = tf.expand_dims(self.sentence_mask, axis=-1) # [N, self.sentence_length, 1]
		encoder_input = encoder_input * mask # except padding

		#embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)

		# stack encoder layer
		for i in range(6):
			# [N, self.sentence_length, self.embedding_size]
			Multihead_add_norm = self.multi_head_attention_add_norm(
					encoder_input, 
					activation=None,
					name='encoder'+str(i)
				) 
			
			# [N, self.sentence_length, self.embedding_size]
			Dense_add_norm = self.dense_add_norm(
					Multihead_add_norm, 
					self.embedding_size, 
					activation=tf.nn.relu,
					name='encoder_dense'+str(i)
				)
		
			encoder_input = Dense_add_norm

		return Dense_add_norm # [N, self.sentence_length, self.embedding_size]


	def train_helper(self, encoder_embedding):
		#encoder_embedding: # [N, self.sentence_length, self.embedding_size]
		
		target_slice = self.target[:, :-1] # [N, self.target_length-1]
		go_input = tf.pad(
				target_slice, 
				[[0,0], [1,0]], 
				'CONSTANT', 
				constant_values=self.go_idx
			) # 왼쪽으로 1칸. [N, self.target_length]
		decoder_input = tf.nn.embedding_lookup(self.output_embedding_table, go_input) # [N, self.target_length, self.embedding_size]
		decoder_input += self.PE[:self.target_length, :]
		#mask = tf.expand_dims(self.target_mask, axis=-1) # [N, self.target_length, 1]
		#decoder_input = decoder_input * mask # except "eos and pad"(== only use "go and valid input")

		# [N, self.target_length, self.voca_size]
		decoder_output = self.decoder(
				decoder_input,
				encoder_embedding
			)

		return decoder_output
		'''
		go_input:
			[[ go 9  1  3  5  6 13(eos) -1 -1 -1 -1]
	 		[ go 4  4  9  9  4 13 -1 -1 -1 -1]
	 		[ go 1  3  4  1  6  2 13 -1 -1 -1]
	 		[ go 1  1  3  9  1 13 -1 -1 -1 -1]
	 		[ go 6  9  1  8  1 13 -1 -1 -1 -1]]
		--------------------------------------------------
		decoder_input: before mask
			go_embedding+PE  go_embedding+PE  go_embedding+PE
			1st_word+PE      1st_word+PE      1st_word+PE
			2nd_word+PE      2nd_word+PE      2nd_word+PE
			eos+PE           eos+PE            eos+PE
			pad+PE           pad+PE            pad+PE

			go_embedding+PE  go_embedding+PE  go_embedding+PE
			1st_word+PE      1st_word+PE      1st_word+PE
			2nd_word+PE      2nd_word+PE      2nd_word+PE
			eos+PE           eos+PE            eos+PE
			pad+PE           pad+PE            pad+PE
		--------------------------------------------------
		decoder_input: after mask (except eos and pad)
	 		go_embedding+PE  go_embedding+PE  go_embedding+PE
			1st_word+PE      1st_word+PE      1st_word+PE
			2nd_word+PE      2nd_word+PE      2nd_word+PE
			0                 0                 0
			0                 0                 0

			go_embedding+PE  go_embedding+PE  go_embedding+PE
			1st_word+PE      1st_word+PE      1st_word+PE
			2nd_word+PE      2nd_word+PE      2nd_word+PE
			0                 0                 0
			0                 0                 0       
		'''


	
	
	def infer_helper(self, encoder_embedding):
		#encoder_embedding: # [N, self.sentence_length, self.embedding_size]

		N = tf.shape(self.sentence)[0] # batchsize
		go_input = tf.one_hot(
				tf.zeros([N], tf.int32), 
				self.target_length, 
				on_value=self.go_idx, 
				off_value=-1
			) # [N, self.target_length]
		decoder_input = tf.nn.embedding_lookup(self.output_embedding_table, go_input) # [N, self.target_length, self.embedding_size]
		decoder_input += self.PE[:self.target_length, :] 
		
		'''
		go_input:
			go_idx -1 -1 -1 -1 -1 -1  # -1은 embedding_lookup하면 0처리되므로.
			go_idx -1 -1 -1 -1 -1 -1
			go_idx -1 -1 -1 -1 -1 -1
		-------------------------------------------------------------
		decoder_input:
			go_embedding+PE  go_embedding+PE  go_embedding+PE
			0+PE             0+PE             0+PE
			0+PE             0+PE             0+PE

			go_embedding+PE  go_embedding+PE  go_embedding+PE
			0+PE             0+PE             0+PE
			0+PE             0+PE             0+PE
		'''

		decoder_output = []
		for index in range(self.target_length): 
			current_decoder_output = self.decoder(
					decoder_input, #decoder_input은 index마다 업데이트됨. 
					encoder_embedding, 
				) # [N, self.target_length, self.voca_size]

			current_decoder_output = current_decoder_output[:, index, :] # [N, self.voca_size]
			current_decoder_output = tf.expand_dims(current_decoder_output, axis=1) # [N, 1, self.voca_size]
			decoder_output.append(current_decoder_output) 
			
			argmax_current_output = tf.argmax(current_decoder_output, axis=-1) # [N, 1]

			#Assign argmax_current_output(index) to decoder_input(index+1)
			if index < self.target_length-1:	
				# [N, self.target_length]
				pad_argmax_current_output = tf.pad( 
						argmax_current_output, 
						[[0,0], [index+1, self.target_length-index-2]], 
						mode='CONSTANT', 
						constant_values=-1
					) 
				
				# [N, self.target_length, self.embedding_size]
				embedding_pad_argmax_current_output = tf.nn.embedding_lookup(
						self.output_embedding_table, 
						pad_argmax_current_output
					) 
				
				# [N, self.target_length, self.embedding_size]
				decoder_input += embedding_pad_argmax_current_output
				
				'''
				if index==0:
					pad_argmax_current_output:  
					-1 				 current_output 				 -1
					-1 				 current_output 				 -1
					---------------------------------------------------
					embedding_pad_argmax_current_output:
					0                0                0
					embedding        embedding        embedding
					0                0                0

					0                0                0
					embedding        embedding        embedding
					0                0                0
					---------------------------------------------------
					new decoder_input:
					go_embedding+PE  go_embedding+PE  go_embedding+PE
					embedding+PE     embedding+PE     embedding+PE
					0+PE             0+PE             0+PE

					go_embedding+PE  go_embedding+PE  go_embedding+PE
					embedding+PE     embedding+PE     embedding+PE
					0+PE             0+PE             0+PE					
				'''
		
		# [N, self.target_length, self.voca_size]
		decoder_output = tf.concat(decoder_output, axis=1) 
		return decoder_output 
	


	def decoder(self, decoder_input, encoder_embedding):
		# decoder_input: [N, self.target_length, self.embedding_size]
		# encoder_embedding: # [N, self.sentence_length, self.embedding_size]
	
		# stack decoder layer
		for i in range(6):
			# Masked self attention
			Masked_Multihead_add_norm = self.multi_head_attention_add_norm(
					decoder_input, 
					mask=True,
					activation=None,
					name='self_attention_decoder'+str(i)
				)
			#return Masked_Multihead_add_norm
			# Encoder Decoder attention
			ED_Multihead_add_norm = self.multi_head_attention_add_norm(
					Masked_Multihead_add_norm, 
					encoder_embedding=encoder_embedding,
					activation=None,
					name='ED_attention_decoder'+str(i)
				) 
			#return ED_Multihead_add_norm
			# [N, self.target_length, self.embedding_size]
			Dense_add_norm = self.dense_add_norm(
					ED_Multihead_add_norm,
					units=self.embedding_size, 
					activation=tf.nn.relu,
					name='decoder_dense'+str(i)
				)
			#return Dense_add_norm
			decoder_input = Dense_add_norm

	
		with tf.variable_scope("decoder_linear", reuse=tf.AUTO_REUSE):
			# [N, self.target_length, self.voca_size]
			linear = tf.layers.dense(Dense_add_norm, self.voca_size, activation=None)			
		return linear # softmax는 cost 구할 때 seq2seq.sequence_loss에서 계산되므로 안함.


	def dense_add_norm(self, embedding, units, activation, name=None):
		# FFN(x) = max(0, x*W1+b1)*W2 + b2
		#변수공유  
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			inner_layer = tf.layers.dense(embedding, units=2048, activation=activation)
			dense = tf.layers.dense(inner_layer, units=units, activation=None)
			dense += embedding
			#return dense
			dense = tf.contrib.layers.layer_norm(dense, begin_norm_axis=2)	
		return dense 


	def multi_head_attention_add_norm(self, embedding, mask=False, encoder_embedding=None, activation=None, name=None):
		#변수공유
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
		
			# layers dense는 배치(N)별로 동일하게 연산됨.	
			# for문으로 8번 돌릴 필요 없이 embedding_size 만큼 만들고 8등분해서 연산하면 됨.
			if encoder_embedding is None: #encoder(multi-head attention) or decoder(masked multi_head_attention)에서 계산할 때.
				# [N, ?, self.embedding_size], ('?' is self.sentence_length or self.target_length)
				V = tf.layers.dense(embedding, units=self.embedding_size, activation=activation, use_bias=False) 
				K = tf.layers.dense(embedding, units=self.embedding_size, activation=activation, use_bias=False) 
				Q = tf.layers.dense(embedding, units=self.embedding_size, activation=activation, use_bias=False)

			else: #decoder(multi_head_attention == encoder decoder attention)에서 계산할 때.
			 	# [N, self.sentence_length, self.embedding_size]
				V = tf.layers.dense(encoder_embedding, units=self.embedding_size, activation=activation, use_bias=False)
				K = tf.layers.dense(encoder_embedding, units=self.embedding_size, activation=activation, use_bias=False)
				# [N, self.target_length, self.embedding_size]
				Q = tf.layers.dense(embedding, units=self.embedding_size, activation=activation, use_bias=False) 


			# linear 결과를 8등분하고 연산에 지장을 주지 않도록 batch화 시킴.
			V = tf.split(value=V, num_or_size_splits=8, axis=-1) #8등분. [N, ?, self.embedding_size/8]이 8개 존재.
			V = tf.concat(V, axis=0) # [8*N, ?, self.embedding_size/8]
			K = tf.split(value=K, num_or_size_splits=8, axis=-1) 
			K = tf.concat(K, axis=0) 
			Q = tf.split(value=Q, num_or_size_splits=8, axis=-1) 
			Q = tf.concat(Q, axis=0) 
			
			# Q * (K.T) and scaling
			score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size/8.0) # [8*N, ?, ?]
		
			# masking
			if mask is True:
				score = score * self.decoder_mask # zero mask
				score = score + ((self.decoder_mask-1) * 1e+10) # -inf mask
				
			softmax = tf.nn.softmax(score, dim=2) # [8*N, ?, ?]
			attention = tf.matmul(softmax, V) # [8*N, ?, self.embedding_size/8]			
			attention = tf.split(value=attention, num_or_size_splits=8, axis=0) # [N, ?, self.embedding_size/8]이 8개 존재.
			concat = tf.concat(attention, axis=-1) # [N, ?, self.embedding_size]

			Multihead = tf.layers.dense(concat, units=self.embedding_size, activation=activation) # [N, ?, self.embedding_size]
			#return Multihead
			Multihead += embedding # add
			#print('same', Multihead)
			#return Multihead
		

			Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2) # [N, ?, self.embedding_size] #here error

			#Multihead2 = tf.contrib.layers.layer_norm(tf.add(Multihead, embedding)) # [N, ?, self.embedding_size] #here error
			#print(Multihead2)
			return Multihead


	
	def positional_encoding(self):
		alpha = 20
		PE = np.zeros([self.target_length + alpha, self.embedding_size])
		for pos in range(self.target_length + alpha): #충분히 크게 만들어두고 slice 해서 쓰자.
			for i in range(self.embedding_size//2): 
				PE[pos, 2*i] = np.sin( pos / np.power(10000, 2*i/self.embedding_size) )
				PE[pos, 2*i+1] = np.cos( pos / np.power(10000, 2*i/self.embedding_size) )
		
		return PE #[self.sentence_length, self.embedding_siz]
	
		
'''
sess = tf.Session()
tt = Transformer(sess, sentence_length=2, target_length=5, voca_size=10, embedding_size=16)
#a = np.array([[1, 1], [0,0]], np.int32)

a = np.array([[8, 5, 3, 3, 3], [8,5, 3, 3, 3]], np.int32)
#a = np.array([[8, 5, 3, 3, 3], [7,4, 3, 3, 3]], np.int32)
sen = np.array([[8, 5], [7,4]], np.int32)
pp = sess.run(tt.train_pred_embedding, {tt.sentence:sen, tt.target:a, tt.sentence_sequence_length:[2, 2], tt.target_sequence_length:[3, 3]})
qq = sess.run(tt.infer_pred_embedding, {tt.sentence:sen, tt.sentence_sequence_length:[2, 2]})
#qq = sess.run(tt.pred, {tt.sentence:a, tt.is_train:False})
print('train_pred\n',pp, '\n')
print('infer_pred\n',qq, '\n')
'''
#print(zz.shape)


	
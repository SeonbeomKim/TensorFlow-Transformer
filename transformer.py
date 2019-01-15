#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization
#https://arxiv.org/abs/1512.00567 Label Smoothing

import tensorflow as tf #version 1.4
import numpy as np
import os
#tf.set_random_seed(787)

class Transformer:
	def __init__(self, sess, voca_size, embedding_size, is_embedding_scale, PE_sequence_length,
					encoder_decoder_stack, multihead_num, go_idx, eos_idx, pad_idx, label_smoothing):
		
		self.sess = sess
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.is_embedding_scale = is_embedding_scale # True or False
		self.PE_sequence_length = PE_sequence_length
		self.encoder_decoder_stack = encoder_decoder_stack
		self.multihead_num = multihead_num
		self.go_idx = go_idx # <'go'> symbol index
		self.eos_idx = eos_idx # <'eos'> symbol index
		self.pad_idx = pad_idx # <'pad'> symbol index
		self.label_smoothing = label_smoothing # if 1.0, then one-hot encooding
		self.PE = tf.convert_to_tensor(self.positional_encoding(), dtype=tf.float32) #[self.target_length + alpha, self.embedding_siz] #slice해서 쓰자.

		
		with tf.name_scope("placeholder"):
			self.lr = tf.placeholder(tf.float32)
			self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input') 
			self.encoder_input_length = tf.shape(self.encoder_input)[1]

			self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input') #'go a b c eos pad'
			self.decoder_input_length = tf.shape(self.decoder_input)[1]

			self.target = tf.placeholder(tf.int32, [None, None], name='target') # 'a b c eos pad pad'
			
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
				# dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE)
			'''
			self.feed_encoder_embedding = tf.placeholder(tf.float32, [None, None, self.embedding_size], name='feed_encoder_embedding') # for inference
			self.beam_width = tf.placeholder(tf.int32, name='beam_width')
			self.time_step = tf.placeholder(tf.int32, name='time_step')
			'''

		# random_uniform으로 바꾸고 cpu로 할당하도록 수정하자.

		with tf.name_scope("embedding_table"):
			with tf.device('/cpu:0'):
				zero = tf.zeros([1, self.embedding_size], dtype=tf.float32) # for padding 
				#embedding_table = tf.Variable(tf.random_uniform([self.voca_size-1, self.embedding_size], -1, 1))
				embedding_table = tf.get_variable( # https://github.com/tensorflow/models/blob/master/official/transformer/model/embedding_layer.py
						'embedding_table', 
						[self.voca_size-1, self.embedding_size], 
						initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5))
				front, end = tf.split(embedding_table, [self.pad_idx, self.voca_size-1-self.pad_idx])
				self.embedding_table = tf.concat((front, zero, end), axis=0) # [self.voca_size, self.embedding_size]


		with tf.name_scope("masks"):
			self.encoder_input_mask = tf.expand_dims(
					tf.cast(tf.not_equal(self.encoder_input, self.pad_idx),	dtype=tf.float32), # [N, encoder_input_length]
					axis=-1
				) # [N, encoder_input_length, 1] 
			
			encoder_multihead_attention_mask = tf.matmul(
					self.encoder_input_mask,
					tf.transpose(self.encoder_input_mask, [0, 2, 1])
				) # [N, encoder_input_length, encoder_input_length]
			self.encoder_multihead_attention_mask = tf.tile(
					encoder_multihead_attention_mask, 
					[self.multihead_num, 1, 1]
				) # [self.multihead_num*N, encoder_input_length, encoder_input_length]
			
			ED_attention_decoder_mask = tf.transpose(self.encoder_input_mask, [0, 2, 1]) # [N, 1, encoder_input_length]
			self.ED_attention_decoder_mask = tf.tile(
					ED_attention_decoder_mask,
					[self.multihead_num, 1, 1]
				) # [self.multihead_num*N, 1, encoder_input_length] # 1 부분은 embedding_size만큼 broadcasting 됨.


			self.decoder_mask = tf.sequence_mask(
					tf.range(start=1, limit=self.decoder_input_length+1), # [start, limit)
					maxlen=self.decoder_input_length,#.eval(session=sess),
					dtype=tf.float32
				) # [decoder_input_length, decoder_input_length]
			self.target_pad_mask = tf.cast( #sequence_mask처럼 생성됨
					tf.not_equal(self.target, self.pad_idx),
					dtype=tf.float32
				) # [N, target_length] (include eos)
	

		with tf.name_scope('encoder'):
			self.encoder_input_embedding = self.embedding_and_PE(self.encoder_input, self.encoder_input_length)
			self.encoder_embedding = self.encoder(self.encoder_input_embedding)

		
		with tf.name_scope('train_decoder'):
			decoder_input_embedding = self.embedding_and_PE(self.decoder_input, self.decoder_input_length) # decoder_input은 go 붙어있어야함.
			self.decoder_embedding, self.decoder_pred = self.decoder(decoder_input_embedding, self.encoder_embedding)
			
		
		#with tf.name_scope('infer_decoder'):
			#self.infer_embedding, self.infer_pred = self.decoder(decoder_input_embedding, self.feed_encoder_embedding)
			# for beam search
			#self.top_k_prob, self.top_k_indices = self.beam_search_graph(self.infer_embedding, self.time_step, self.beam_width)
		

		with tf.name_scope('train_cost'): 
			# make smoothing target one hot vector
			self.target_one_hot = tf.one_hot(
					self.target, 
					depth=self.voca_size,
					on_value = (1.0-self.label_smoothing) + (self.label_smoothing / self.voca_size), # tf.float32
					off_value = (self.label_smoothing / self.voca_size), # tf.float32
					dtype= tf.float32
				) # [N, self.target_length, self.voca_size]
			
			# calc train_cost
			self.train_cost = tf.nn.softmax_cross_entropy_with_logits(
					labels = self.target_one_hot, 
					logits = self.decoder_embedding
				) # [N, self.target_length]
			self.train_cost *= self.target_pad_mask # except pad
			self.train_cost = tf.reduce_sum(self.train_cost) / tf.reduce_sum(self.target_pad_mask)

		
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9) 
			self.minimize = optimizer.minimize(self.train_cost)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)
		
		sess.run(tf.global_variables_initializer())


	'''
	def beam_search_graph(self, infer_embedding, time_step, beam_width):
		# infer_embeding: [N*beam_width, decoder_input_length, self.voca_size]
		
		# get current output
		infer_embedding = infer_embedding[:, time_step, :] # [N*beam_width, self.voca_size]
		softmax_infer_embedding = tf.nn.softmax(infer_embedding, dim=-1)

		# get top_k(beam_size) argmax prob, indices
		top_k_prob, top_k_indices = tf.nn.top_k(
				softmax_infer_embedding, # [N*beam_width, self.voca_size]
				beam_width
			) # [N*beam_width, beam_width], [N*beam_width, beam_width]

		top_k_prob = tf.log(tf.reshape(top_k_prob, [-1, 1])) # [N*beam_width*beam_width, 1]
		top_k_indices = tf.reshape(top_k_indices, [-1, 1]) # [N*beam_width*beam_width, 1]

		return top_k_prob, top_k_indices#, softmax_infer_embedding
	'''


	def embedding_and_PE(self, data, data_length):
		# embedding lookup and scale
		with tf.device('/cpu:0'):
			embedding = tf.nn.embedding_lookup(
					self.embedding_table, 
					data
				) # [N, self.data_length, self.embedding_size]		
		if self.is_embedding_scale is True:
			embedding *= self.embedding_size**0.5
		# Add Position Encoding
		embedding += self.PE[:data_length, :]
		# Drop out
		embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)

		return embedding



	def encoder(self, encoder_input_embedding):
		# encoder_input_embedding masking(pad position)
		encoder_input_embedding *= self.encoder_input_mask
		
		# stack encoder layer
		for i in range(self.encoder_decoder_stack): #6
			# Multi-Head Attention
			Multihead_add_norm = self.multi_head_attention_add_norm(
					query=encoder_input_embedding,
					key_value=encoder_input_embedding,
					mask=self.encoder_multihead_attention_mask,
					activation=None,
					name='encoder'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]
		
			# Feed Forward
			encoder_input_embedding = self.dense_add_norm(
					Multihead_add_norm, 
					self.embedding_size, 
					activation=tf.nn.relu,
					name='encoder_dense'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]
			
			# encoder_input_embedding masking(pad position)
			encoder_input_embedding *= self.encoder_input_mask

		return encoder_input_embedding # [N, self.encoder_input_length, self.embedding_size]



	def decoder(self, decoder_input_embedding, encoder_embedding):
		# decoder_input_embedding: [N, self.decoder_input_length, self.embedding_size]
		# encoder_embedding: [N, self.encoder_input_length, self.embedding_size]
		
		# stack decoder layer
		for i in range(self.encoder_decoder_stack):
			# Masked Multi-Head Attention
			Masked_Multihead_add_norm = self.multi_head_attention_add_norm(
					query=decoder_input_embedding, 
					key_value=decoder_input_embedding,
					mask=self.decoder_mask,
					activation=None,
					name='self_attention_decoder'+str(i)
				)
			
			# Multi-Head Attention(Encoder Decoder Attention)
			ED_Multihead_add_norm = self.multi_head_attention_add_norm(
					query=Masked_Multihead_add_norm, 
					key_value=encoder_embedding,
					mask=self.ED_attention_decoder_mask,
					activation=None,
					name='ED_attention_decoder'+str(i)
				) 

			#Feed Forward
			decoder_input_embedding = self.dense_add_norm(
					ED_Multihead_add_norm,
					units=self.embedding_size, 
					activation=tf.nn.relu,
					name='decoder_dense'+str(i)
				) # [N, self.decoder_input_length, self.embedding_size]

		'''
		with tf.variable_scope("decoder_linear", reuse=tf.AUTO_REUSE):
			decoder_embedding = tf.layers.dense(
					decoder_input_embedding,#Dense_add_norm, 
					self.voca_size, 
					activation=None,
				) # [N, self.decoder_input_length, self.voca_size]
		'''

		# share weight, input embeddings, per-softmax layer
		decoder_embedding = tf.reshape(
				decoder_input_embedding, 
				[-1, self.embedding_size]
			) # [N*self.decoder_input_length, self.embedding_size]
		decoder_embedding = tf.matmul(
				decoder_embedding,
				tf.transpose(self.embedding_table) # [self.embedding_size, self.voca_size]
			) # [N*self.decoder_input_length, self.voca_size]
		decoder_embedding = tf.reshape(
				decoder_embedding,
				[-1, self.decoder_input_length, self.voca_size]
			)				

		decoder_pred = tf.argmax(
				decoder_embedding, 
				axis=-1, 
				output_type=tf.int32
			) # [N, self,decoder_input_length]
		
		return decoder_embedding, decoder_pred



	def multi_head_attention_add_norm(self, query, key_value, mask=None, activation=None, name=None):
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# for문으로 self.multihead_num번 돌릴 필요 없이 embedding_size 만큼 만들고 self.multihead_num등분해서 연산하면 됨.	
			V = tf.layers.dense( # layers dense는 배치(N)별로 동일하게 연산됨.	
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False
				) # [N, key_value_sequence_length, self.embedding_size]
			K = tf.layers.dense(
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False
				) # [N, key_value_sequence_length, self.embedding_size]
			Q = tf.layers.dense(
					query, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False
				) # [N, query_sequence_length, self.embedding_size]

			# linear 결과를 self.multihead_num등분하고 연산에 지장을 주지 않도록 batch화 시킴.
			# https://github.com/Kyubyong/transformer 참고.
			# split: [N, key_value_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재 
			V = tf.concat(tf.split(V, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
			K = tf.concat(tf.split(K, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, key_value_sequence_length, self.embedding_size/self.multihead_num]
			Q = tf.concat(tf.split(Q, self.multihead_num, axis=-1), axis=0) # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]
	
			
			# Q * (K.T) and scaling ,  [self.multihead_num*N, query_sequence_length, key_value_sequence_length]
			score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size/self.multihead_num) 

			# masking
			if mask is not None:
				score = score * mask # zero mask
				score = score + ((mask-1) * 1e+9) # -inf mask
				# decoder mask:
				# 1 0 0
				# 1 1 0
				# 1 1 1 형태로 마스킹

				# encoder_multihead_mask
				# 1 1 0
				# 1 1 0
				# 0 0 0 형태로 마스킹

				# ED_attention_mask
				# 1 1 0
				# 1 1 0 
				# 1 1 0 형태로 마스킹
			
			softmax = tf.nn.softmax(score, dim=2) # [self.multihead_num*N, query_sequence_length, key_value_sequence_length]
		
			if mask is not None:
				softmax = softmax * mask # zero mask

			# Attention dropout
			# https://arxiv.org/abs/1706.03762v4 => v4 paper에는 attention dropout 하라고 되어 있음. 
			softmax = tf.nn.dropout(softmax, keep_prob=self.keep_prob)
			
			# Attention weighted sum
			attention = tf.matmul(softmax, V) # [self.multihead_num*N, query_sequence_length, self.embedding_size/self.multihead_num]			

			# split: [N, query_sequence_length, self.embedding_size/self.multihead_num]이 self.multihead_num개 존재
			concat = tf.concat(tf.split(attention, self.multihead_num, axis=0), axis=-1) # [N, query_sequence_length, self.embedding_size]
			
			# Linear
			Multihead = tf.layers.dense(
					concat, 
					units=self.embedding_size, 
					activation=activation,
					use_bias=False
				) # [N, query_sequence_length, self.embedding_size]
			# residual Drop Out
			Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
			# Add
			Multihead += query
			# Layer Norm			
			Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2) # [N, query_sequence_length, self.embedding_size]
			
			return Multihead



	def dense_add_norm(self, embedding, units, activation, name=None):
		# FFN(x) = max(0, x*W1+b1)*W2 + b2
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# FFN
			'''
			dense = tf.layers.conv1d(
					inputs=embedding,
					filters=4*self.embedding_size,
					kernel_size=1,
					strides=1,
					activation=activation #relu
				)
			dense = tf.layers.conv1d(
					inputs=dense,
					filters=self.embedding_size,
					kernel_size=1,
					strides=1,
					activation=None
				)
			'''
			
			inner_layer = tf.layers.dense(
					embedding, 
					units=4*self.embedding_size, #bert paper 
					activation=activation # relu
				) # [N, self.decoder_input_length, 4*self.embedding_size]
			dense = tf.layers.dense(
					inner_layer, 
					units=units, 
					activation=None
				) # [N, self.decoder_input_length, self.embedding_size]
			
			# Drop out 
			dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
			# Add
			dense += embedding			
			# Layer Norm
			dense = tf.contrib.layers.layer_norm(dense,	begin_norm_axis=2)
	
		return dense 
	
	def positional_encoding(self):
		PE = np.zeros([self.PE_sequence_length, self.embedding_size], np.float32)
		for pos in range(self.PE_sequence_length): #충분히 크게 만들어두고 slice 해서 쓰자.
			sin, cos = [], []
			for i in range(0, self.embedding_size//2): 
				sin.append(np.sin( pos / np.power(10000, 2*i/self.embedding_size) ).astype(np.float32))
				cos.append(np.cos( pos / np.power(10000, 2*i/self.embedding_size) ).astype(np.float32))
			PE[pos] = np.concatenate((sin,cos))
		return PE #[self.PE_sequence_length, self.embedding_siz]
	
	'''
	# 기존
	def positional_encoding(self):
		PE = np.zeros([self.PE_sequence_length, self.embedding_size])
		for pos in range(self.PE_sequence_length): #충분히 크게 만들어두고 slice 해서 쓰자.
			for i in range(self.embedding_size//2): 
				PE[pos, 2*i] = np.sin( pos / np.power(10000, 2*i/self.embedding_size) )
				PE[pos, 2*i+1] = np.cos( pos / np.power(10000, 2*i/self.embedding_size) )
		
		return PE #[self.PE_sequence_length, self.embedding_siz]
	'''
#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization
#https://arxiv.org/abs/1512.00567 Label Smoothing

import tensorflow as tf #version 1.4
import numpy as np
import os
#tf.set_random_seed(787)

class Transformer:
	def __init__(self, sess, voca_size, embedding_size, is_embedding_scale, PE_sequence_length,
					encoder_decoder_stack, multihead_num, eos_idx, pad_idx, label_smoothing):
		
		self.sess = sess
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.is_embedding_scale = is_embedding_scale # True or False
		self.PE_sequence_length = PE_sequence_length
		self.encoder_decoder_stack = encoder_decoder_stack
		self.multihead_num = multihead_num
		self.eos_idx = eos_idx # <'eos'> symbol index
		self.pad_idx = pad_idx # <'pad'> symbol index
		self.label_smoothing = label_smoothing # if 1.0, then one-hot encooding
		self.PE = tf.convert_to_tensor(self.positional_encoding(), dtype=tf.float32) #[self.PE_sequence_length, self.embedding_siz] #slice해서 쓰자.

		
		with tf.name_scope("placeholder"):
			self.lr = tf.placeholder(tf.float32)
			self.encoder_input = tf.placeholder(tf.int32, [None, None], name='encoder_input') 
			self.encoder_input_length = tf.shape(self.encoder_input)[1]

			self.decoder_input = tf.placeholder(tf.int32, [None, None], name='decoder_input') #'go a b c eos pad'
			self.decoder_input_length = tf.shape(self.decoder_input)[1]

			self.target = tf.placeholder(tf.int32, [None, None], name='target') # 'a b c eos pad pad'
			
			self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
				# dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE) and (attention)
	

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


		with tf.name_scope('encoder'):
			encoder_input_embedding, encoder_input_mask = self.embedding_and_PE(self.encoder_input, self.encoder_input_length)
			self.encoder_embedding = self.encoder(encoder_input_embedding, encoder_input_mask)

		
		with tf.name_scope('decoder'):
			decoder_input_embedding, decoder_input_mask = self.embedding_and_PE(self.decoder_input, self.decoder_input_length) # decoder_input은 go 붙어있어야함.
			self.decoder_embedding, self.decoder_pred = self.decoder(decoder_input_embedding, self.encoder_embedding, decoder_input_mask)
			

		with tf.name_scope('train_cost'): 
			# target mask ( masking pad of target )
			self.target_pad_mask = tf.cast( #sequence_mask처럼 생성됨
					tf.not_equal(self.target, self.pad_idx),
					dtype=tf.float32
				) # [N, target_length] (include eos)

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



	def embedding_and_PE(self, data, data_length):
		# data: [N, data_length]

		# embedding lookup and scale
		with tf.device('/cpu:0'):
			embedding = tf.nn.embedding_lookup(
					self.embedding_table, 
					data
				) # [N, data_length, self.embedding_size]		
		if self.is_embedding_scale is True:
			embedding *= self.embedding_size**0.5

		embedding_mask = tf.expand_dims(
				tf.cast(tf.not_equal(data, self.pad_idx), dtype=tf.float32), # [N, data_length]
				axis=-1
			) # [N, data_length, 1] 

		# Add Position Encoding
		embedding += self.PE[:data_length, :]

		# pad masking (set 0 PE added pad position)
		embedding *= embedding_mask

		# Drop out
		embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)
		return embedding, embedding_mask



	def encoder(self, encoder_input_embedding, encoder_input_mask):
		# encoder_input_embedding: [N, self.encoder_input_length, self.embedding_size] , pad mask applied
		# encoder_input_mask: [N, self.encoder_input_length, 1]
		
		# mask
		encoder_self_attention_mask = tf.tile(
				tf.matmul(encoder_input_mask, tf.transpose(encoder_input_mask, [0, 2, 1])), # [N, encoder_input_length, encoder_input_length] 
				[self.multihead_num, 1, 1]
			) # [self.multihead_num*N, encoder_input_length, encoder_input_length]

		for i in range(self.encoder_decoder_stack): #6
			# Multi-Head Attention
			Multihead_add_norm = self.multi_head_attention_add_norm(
					query=encoder_input_embedding,
					key_value=encoder_input_embedding,
					score_mask=encoder_self_attention_mask,
					output_mask=encoder_input_mask,
					activation=None,
					name='encoder'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]
		
			# Feed Forward
			encoder_input_embedding = self.dense_add_norm(
					Multihead_add_norm, 
					self.embedding_size,
					output_mask=encoder_input_mask, # set 0 bias added pad position
					activation=tf.nn.relu,
					name='encoder_dense'+str(i)
				) # [N, self.encoder_input_length, self.embedding_size]

		return encoder_input_embedding # [N, self.encoder_input_length, self.embedding_size]



	def decoder(self, decoder_input_embedding, encoder_embedding, decoder_input_mask):
		# decoder_input_embedding: [N, self.decoder_input_length, self.embedding_size] , pad mask applied
		# encoder_embedding: [N, self.encoder_input_length, self.embedding_size] , pad mask applied
		# decoder_input_mask: [N, self.decoder_input_length, 1]
		
		# mask
		pad_of_encoder_embedding = tf.transpose(
				tf.reduce_sum(tf.abs(encoder_embedding), axis=-1, keep_dims=True), # [N, self.encoder_input_length, 1]
				[0, 2, 1] 
			) # [N, 1, encoder_input_length]
		decoder_ED_attention_mask = tf.tile(
				tf.cast(tf.not_equal(pad_of_encoder_embedding, self.pad_idx), dtype=tf.float32), # [N, 1, encoder_input_length]
				[self.multihead_num, 1, 1]
			) # [self.multihead_num*N, 1, encoder_input_length]
		decoder_self_attention_mask = tf.sequence_mask(
				tf.range(start=1, limit=self.decoder_input_length+1), # [start, limit)
				maxlen=self.decoder_input_length,#.eval(session=sess),
				dtype=tf.float32
			) # [decoder_input_length, decoder_input_length]


		for i in range(self.encoder_decoder_stack):
			# Masked Multi-Head Attention
			Masked_Multihead_add_norm = self.multi_head_attention_add_norm(
					query=decoder_input_embedding, 
					key_value=decoder_input_embedding,
					score_mask=decoder_self_attention_mask,
					output_mask=decoder_input_mask, 
					activation=None,
					name='self_attention_decoder'+str(i)
				)
			
			# Multi-Head Attention(Encoder Decoder Attention)
			ED_Multihead_add_norm = self.multi_head_attention_add_norm(
					query=Masked_Multihead_add_norm, 
					key_value=encoder_embedding,
					score_mask=decoder_ED_attention_mask,
					output_mask=decoder_input_mask, 
					activation=None,
					name='ED_attention_decoder'+str(i)
				) 

			#Feed Forward
			decoder_input_embedding = self.dense_add_norm(
					ED_Multihead_add_norm,
					units=self.embedding_size, 
					output_mask=decoder_input_mask, # set 0 bias added pad position
					activation=tf.nn.relu,
					name='decoder_dense'+str(i)
				) # [N, self.decoder_input_length, self.embedding_size]


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
			) # [N, self.decoder_input_length, self.voca_size]	

		decoder_pred = tf.argmax(
				decoder_embedding, 
				axis=-1, 
				output_type=tf.int32
			) # [N, self.decoder_input_length]
		
		return decoder_embedding, decoder_pred



	def multi_head_attention_add_norm(self, query, key_value, score_mask=None, output_mask=None, activation=None, name=None):
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# for문으로 self.multihead_num번 돌릴 필요 없이 embedding_size 만큼 만들고 self.multihead_num등분해서 연산하면 됨.	
			V = tf.layers.dense( # layers dense는 배치(N)별로 동일하게 연산됨.	
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='V'
				) # [N, key_value_sequence_length, self.embedding_size]
			K = tf.layers.dense(
					key_value, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='K'
				) # [N, key_value_sequence_length, self.embedding_size]
			Q = tf.layers.dense(
					query, 
					units=self.embedding_size, 
					activation=activation, 
					use_bias=False,
					name='Q'
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
			if score_mask is not None:
				score *= score_mask # zero mask
				score += ((score_mask-1) * 1e+9) # -inf mask
				# decoder self_attention:
				# 1 0 0
				# 1 1 0
				# 1 1 1 형태로 마스킹

				# encoder_self_attention
				# if encoder_input_data: i like </pad>
				# 1 1 0
				# 1 1 0
				# 0 0 0 형태로 마스킹

				# ED_attention
				# if encoder_input_data: i like </pad>
				# 1 1 0
				# 1 1 0 
				# 1 1 0 형태로 마스킹
			
			softmax = tf.nn.softmax(score, dim=2) # [self.multihead_num*N, query_sequence_length, key_value_sequence_length]

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
					use_bias=False,
					name='linear'
				) # [N, query_sequence_length, self.embedding_size]

			if output_mask is not None:
				Multihead *= output_mask

			# residual Drop Out
			Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)
			# Add
			Multihead += query
			# Layer Norm			
			Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2) # [N, query_sequence_length, self.embedding_size]
			
			return Multihead



	def dense_add_norm(self, embedding, units, output_mask=None, activation=None, name=None):
		# FFN(x) = max(0, x*W1+b1)*W2 + b2
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):		
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
			
			if output_mask is not None:
				dense *= output_mask # set 0 bias added pad position

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
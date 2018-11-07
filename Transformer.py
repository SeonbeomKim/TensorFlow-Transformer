#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization
#https://arxiv.org/abs/1512.00567 Label Smoothing

import tensorflow as tf #version 1.4
import numpy as np
import os

# tf.set_random_seed(777)

class Transformer:
	def __init__(self, sess, voca_size, embedding_size, is_embedding_scale, PE_sequence_length,
					encoder_decoder_stack, go_idx, eos_idx, pad_idx, label_smoothing):
		
		self.sess = sess
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.is_embedding_scale = is_embedding_scale # True or False
		self.PE_sequence_length = PE_sequence_length
		self.encoder_decoder_stack = encoder_decoder_stack
		self.go_idx = go_idx # <'go'> symbol index
		self.eos_idx = eos_idx # <'eos'> symbol index
		self.pad_idx = pad_idx # -1
		self.label_smoothing = 0.1 # if 1.0, then one-hot encooding
		self.PE = tf.convert_to_tensor(self.positional_encoding(), dtype=tf.float32) #[self.target_length + alpha, self.embedding_siz] #slice해서 쓰자.
		#self.lr = 0.0001

		
		with tf.name_scope("placeholder"):
			self.lr = tf.placeholder(tf.float32)
			self.sentence = tf.placeholder(tf.int32, [None, None]) 
			self.target = tf.placeholder(tf.int32, [None, None])
			self.sentence_length = tf.shape(self.sentence)[1]
			self.target_length = tf.shape(self.target)[1]
			self.keep_prob = tf.placeholder(tf.float32) 
				# dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE)
		

		with tf.name_scope("embedding_table"):
			self.embedding_table = tf.Variable(	tf.random_normal([self.voca_size, self.embedding_size])	) 


		with tf.name_scope("masks"):
			self.encoder_input_mask = tf.cast(
						tf.not_equal(self.sentence, self.pad_idx),
						dtype=tf.float32
					)

			self.decoder_mask = tf.sequence_mask(
						#tf.constant([i for i in range(1, 20+1)]),
						tf.range(start=1, limit=self.target_length+1),
						maxlen=self.target_length,#.eval(session=sess),
						dtype=tf.float32
					) # [target_length, target_length]
			
			self.target_pad_mask = tf.cast( #sequence_mask처럼 생성됨.
						tf.not_equal(self.target, self.pad_idx),
						dtype=tf.float32
					) # [N, target_length] (include eos)


		

		with tf.name_scope('encoder'):
			self.encoder_embedding = self.encoder() # [N, self.sentence_length, self.embedding_size]


		with tf.name_scope('train_decoder'):	
			# self.train_pred: [N, self.target_length], self.train_embedding: [N, self.target_length, self.voca_size]
			self.train_pred, self.train_embedding = self.train_helper(self.encoder_embedding) 
			
			# train_output masking(remove eos, pad)
			self.train_first_eos = tf.argmax(
						tf.cast(tf.equal(self.train_pred, self.eos_idx), tf.int32),
						axis=-1
					) # [N],  find first eos index  ex [5, 6, 4, 5, 5]
			self.train_eos_mask = tf.sequence_mask(
						self.train_first_eos,
						maxlen=self.target_length,
						dtype=tf.int32
					)
			self.train_pred_except_eos = self.train_pred * self.train_eos_mask
			self.train_pred_except_eos += (self.train_eos_mask - 1) # the value of the masked position is -1

		
		with tf.name_scope('train_cost'): 
			# make smoothing target one hot vector
			target_one_hot = tf.one_hot(
						self.target, # [None, self.target_length]
						depth=self.voca_size,
						on_value = 1., # tf.float32
						off_value = 0., # tf.float32
					) # [N, self.target_length, self.voca_size]
			self.smoothing_target_one_hot = target_one_hot * (1-self.label_smoothing) + (self.label_smoothing / self.voca_size)
			
			# calc train_cost
			self.train_cost = tf.nn.softmax_cross_entropy_with_logits(
						labels = self.smoothing_target_one_hot, 
						logits = self.train_embedding
					) # [N, self.target_length]
			self.train_cost = self.train_cost * self.target_pad_mask # except pad
			self.train_cost = tf.reduce_sum(self.train_cost) / tf.reduce_sum(self.target_pad_mask)

		
		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9) 
			self.minimize = optimizer.minimize(self.train_cost)


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)
	

		sess.run(tf.global_variables_initializer())



	def train_helper(self, encoder_embedding):
		#encoder_embedding: [N, self.sentence_length, self.embedding_size]
		
		# decoder input preprocessing
		target_slice = self.target[:, :-1] # [N, self.target_length-1]
		go_input = tf.pad(
					target_slice, 
					[[0,0], [1,0]],  # left side
					'CONSTANT', 
					constant_values=self.go_idx
				) # [N, self.target_length]

		# embedding lookup and scale
		decoder_input = tf.nn.embedding_lookup(
					self.embedding_table, 
					go_input
				) # [N, self.target_length, self.embedding_size]
		if self.is_embedding_scale is True:
			decoder_input *= self.embedding_size**0.5
		# Add Position Encoding
		decoder_input += self.PE[:self.target_length, :]
		# Drop out 
		decoder_input = tf.nn.dropout(decoder_input, keep_prob=self.keep_prob)


		# decoding
		decoder_output = self.decoder(
					decoder_input,
					encoder_embedding
				) # [N, self.target_length, self.voca_size]

		result = tf.argmax(
					decoder_output, 
					axis=-1, 
					output_type=tf.int32
				) # [N, self,target_length]
				
		return result, decoder_output
		'''
		go_input:
			go_idx 1st_word 2nd_word eos pad
	 		go_idx 1st_word 2nd_word eos pad
	 		go_idx 1st_word 2nd_word eos pad
	 		go_idx 1st_word 2nd_word eos pad
	 		go_idx 1st_word 2nd_word eos pad
		--------------------------------------------------
		decoder_input: (The output of eos+PE, pad+PE are ignored when calculating the loss.)
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
		'''
		

	def encoder(self):
		# embedding lookup and scale
		encoder_input = tf.nn.embedding_lookup(
					self.embedding_table,
					self.sentence
				) # [N, self.sentence_length, self.embedding_size]
		
		if self.is_embedding_scale is True:
			encoder_input *= self.embedding_size**0.5
		# Add Position Encoding
		encoder_input += self.PE[:self.sentence_length, :] 
		# Drop out 
		encoder_input = tf.nn.dropout(encoder_input, keep_prob=self.keep_prob)
	
		# test
		encoder_input *= tf.expand_dims(self.encoder_input_mask, axis=-1)

		# stack encoder layer
		for i in range(self.encoder_decoder_stack): #6
			# Multi-Head Attention
			Multihead_add_norm = self.multi_head_attention_add_norm(
						query=encoder_input,
						key_value=encoder_input,
						activation=None,
						name='encoder'+str(i)
					) # [N, self.sentence_length, self.embedding_size]

			# Feed Forward
			Dense_add_norm = self.dense_add_norm(
						Multihead_add_norm, 
						self.embedding_size, 
						activation=tf.nn.relu,
						name='encoder_dense'+str(i)
					) # [N, self.sentence_length, self.embedding_size]

			#######
			#test
			Dense_add_norm *= tf.expand_dims(self.encoder_input_mask, axis=-1)
			#######
	
			encoder_input = Dense_add_norm

		return Dense_add_norm # [N, self.sentence_length, self.embedding_size]



	def decoder(self, decoder_input, encoder_embedding):
		# decoder_input: [N, self.target_length, self.embedding_size]
		# encoder_embedding: [N, self.sentence_length, self.embedding_size]
		
		# stack decoder layer
		for i in range(self.encoder_decoder_stack):
			# Masked Multi-Head Attention
			Masked_Multihead_add_norm = self.multi_head_attention_add_norm(
						query=decoder_input, 
						key_value=decoder_input,
						mask=self.decoder_mask,
						activation=None,
						name='self_attention_decoder'+str(i)
					)
			# Multi-Head Attention(Encoder Decoder Attention)
			ED_Multihead_add_norm = self.multi_head_attention_add_norm(
						query=Masked_Multihead_add_norm, 
						key_value=encoder_embedding,
						activation=None,
						name='ED_attention_decoder'+str(i)
					) 
			#Feed Forward
			Dense_add_norm = self.dense_add_norm(
						ED_Multihead_add_norm,
						units=self.embedding_size, 
						activation=tf.nn.relu,
						name='decoder_dense'+str(i)
					) # [N, self.target_length, self.embedding_size]
			decoder_input = Dense_add_norm

		with tf.variable_scope("decoder_linear", reuse=tf.AUTO_REUSE):
			linear = tf.layers.dense(
						Dense_add_norm, 
						self.voca_size, 
						activation=None
					) # [N, self.target_length, self.voca_size]

		return linear



	def multi_head_attention_add_norm(self, query, key_value, mask=None, activation=None, name=None):
		# Sharing Variables
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
	
			# for문으로 8번 돌릴 필요 없이 embedding_size 만큼 만들고 8등분해서 연산하면 됨.	
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
			
			# linear 결과를 8등분하고 연산에 지장을 주지 않도록 batch화 시킴.
			# https://github.com/Kyubyong/transformer 참고.
			# split: [N, key_value_sequence_length, self.embedding_size/8]이 8개 존재 
			V = tf.concat(tf.split(V, 8, axis=-1), axis=0) # [8*N, key_value_sequence_length, self.embedding_size/8]
			K = tf.concat(tf.split(K, 8, axis=-1), axis=0) # [8*N, key_value_sequence_length, self.embedding_size/8]
			Q = tf.concat(tf.split(Q, 8, axis=-1), axis=0) # [8*N, query_sequence_length, self.embedding_size/8]
			
			# Q * (K.T) and scaling ,  [8*N, query_sequence_length, key_value_sequence_length]
			score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size/8.0) 
			#return score

			# masking
			if mask is not None:
				score = score * mask # zero mask
				score = score + ((mask-1) * 1e+10) # -inf mask

			###
			# test
			else:
				encoder_score_mask = tf.cast(
							tf.not_equal(score, 0.),
							dtype=tf.float32
						)
				score = score * encoder_score_mask # zero mask
				score = score + ((encoder_score_mask-1) * 1e+10) # -inf mask
			####

			softmax = tf.nn.softmax(score, dim=2) # [8*N, query_sequence_length, key_value_sequence_length]
			attention = tf.matmul(softmax, V) # [8*N, query_sequence_length, self.embedding_size/8]			
			

			# split: [N, query_sequence_length, self.embedding_size/8]이 8개 존재
			concat = tf.concat(tf.split(attention, 8, axis=0), axis=-1) # [N, query_sequence_length, self.embedding_size]

			# Linear
			Multihead = tf.layers.dense(
						concat, 
						units=self.embedding_size, 
						activation=activation
					) # [N, query_sequence_length, self.embedding_size]

			# Drop Out 
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
			inner_layer = tf.layers.dense(
					embedding, 
					units=4*self.embedding_size, #bert paper 
					activation=activation # relu
				) # [N, self.target_length, units]
			dense = tf.layers.dense(
					inner_layer, 
					units=units, 
					activation=None
				) # [N, self.target_length, self.embedding_size]
			
			# Drop out 
			dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)
			# Add
			dense += embedding			
			# Layer Norm
			dense = tf.contrib.layers.layer_norm(dense,	begin_norm_axis=2)
	
		return dense 


	
	def positional_encoding(self):
		PE = np.zeros([self.PE_sequence_length, self.embedding_size])
		for pos in range(self.PE_sequence_length): #충분히 크게 만들어두고 slice 해서 쓰자.
			for i in range(self.embedding_size//2): 
				PE[pos, 2*i] = np.sin( pos / np.power(10000, 2*i/self.embedding_size) )
				PE[pos, 2*i+1] = np.cos( pos / np.power(10000, 2*i/self.embedding_size) )
		
		return PE #[self.sentence_length, self.embedding_siz]
	

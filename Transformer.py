#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization

import tensorflow as tf #version 1.4
import numpy as np
import os

#tf.set_random_seed(777)

class Transformer:
	def __init__(self, sess, sentence_length, target_length, voca_size, 
					embedding_size, embedding_scale, go_idx, eos_idx, lr, infer_helper):
		
		self.sess = sess
		self.sentence_length = sentence_length #encoder
		self.target_length = target_length #decoder (include eos)
		self.voca_size = voca_size
		self.embedding_size = embedding_size
		self.embedding_scale = embedding_scale # True or False
		self.go_idx = go_idx # <'go'> symbol index
		self.eos_idx = eos_idx # <'eos'> symbol index
		self.lr = lr
		self.PE = self.positional_encoding() #[self.target_length + alpha, self.embedding_siz] #slice해서 쓰자.
		self.infer_helper = infer_helper

		with tf.name_scope("placeholder"):	
			self.sentence = tf.placeholder(tf.int32, [None, self.sentence_length]) 
			self.sentence_sequence_length = tf.placeholder(tf.int32, [None]) # no padded length
			self.target = tf.placeholder(tf.int32, [None, self.target_length])
			self.target_sequence_length = tf.placeholder(tf.int32, [None])  # include eos
			# dropout (each sublayers before add and norm)  and  (sums of the embeddings and the PE)
			self.keep_prob = tf.placeholder(tf.float32) 
			
		with tf.name_scope('masks'): 
			# https://www.tensorflow.org/api_docs/python/tf/sequence_mask
			self.sentence_mask =  tf.sequence_mask(
						self.sentence_sequence_length, 
						maxlen=self.sentence_length, 
						dtype=tf.float32
					) # [N, self.sentence_length] 

			self.target_mask = tf.sequence_mask(
						self.target_sequence_length, 
						maxlen=self.target_length, 
						dtype=tf.float32
					) # [N, target_sequence_length] (include eos)

			self.decoder_mask = tf.sequence_mask(
						tf.constant([i for i in range(1, self.target_length+1)]),
						maxlen=self.target_length,
						dtype=tf.float32
					) # [target_sequence_length, target_sequence_length]


		with tf.name_scope("embedding_table"):
			self.embedding_table = tf.Variable(	tf.random_normal([self.voca_size, self.embedding_size])	) 


		with tf.name_scope('encoder'):
			self.encoder_embedding = self.encoder() # [N, self.sentence_length, self.embedding_size]


		with tf.name_scope('decoder'):
			with tf.name_scope('train'):
				# train_output pred
				self.train_pred_embedding = self.train_helper(self.encoder_embedding) # [N, self.target_length, self.voca_size]
				self.train_pred = tf.argmax(
							self.train_pred_embedding, 
							axis=-1, 
							output_type=tf.int32
						) # [N, self,target_length]
				
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

			with tf.name_scope('inference'):
				# inference_output pred
				self.infer_pred, self.infer_embedding = self.infer_helper.decode( 
							decoder_fn = self.decoder, 
							encoder_embedding = self.encoder_embedding, 
							target_length = self.target_length, 
							PE = self.PE, 
							go_idx = self.go_idx,
							embedding_table = self.embedding_table,
							embedding_scale = self.embedding_scale # True or False
						)# infer_pred: [N, self.target_length], infer_embedding: [N, self.target_length, self.voca_size]
							
				# inference_output masking(remove eos, pad)
				self.infer_first_eos = tf.argmax(
							tf.cast( tf.equal(self.infer_pred, self.eos_idx), tf.int32), 
							axis=-1
						) # [N]
				self.infer_eos_mask = tf.sequence_mask(
							self.infer_first_eos,
							maxlen=self.target_length,
							dtype=tf.int32
						)
				self.infer_pred_except_eos = self.infer_pred * self.infer_eos_mask
				self.infer_pred_except_eos += (self.infer_eos_mask - 1) # the value of the masked position is -1
			

		with tf.name_scope('cost'): 
			# https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss #내부에서 weighted softmax cross entropy 동작.
			self.train_cost = tf.contrib.seq2seq.sequence_loss(
						self.train_pred_embedding, 
						self.target, 
						self.target_mask
					)
			self.infer_cost = tf.contrib.seq2seq.sequence_loss(
						self.infer_embedding, 
						self.target, 
						self.target_mask
					) 


		with tf.name_scope('optimizer'):
			optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9) 
			self.minimize = optimizer.minimize(self.train_cost)


		with tf.name_scope('correct_check'):
			# target masking(remove eos, pad)
			target_eos_mask =  tf.sequence_mask( 
						self.target_sequence_length - 1,  #except eos
						maxlen=self.target_length, 
						dtype=tf.int32
					) 
			target_except_eos = self.target * target_eos_mask # masking eos, pad
			target_except_eos += (target_eos_mask - 1) # the value of the masked position is -1
			
			# correct check
			check_equal_position = tf.cast(
						tf.equal(target_except_eos, self.infer_pred_except_eos), 
						dtype=tf.float32
					) # [N, self.target_length]
		
			check_equal_position_sum = tf.reduce_sum( 	#if use mean, 0.9999999 is equal to 1, so use sum.
						check_equal_position, 
						axis=-1
					) # [N]
			
			correct_check = tf.cast( #if correct: "check_equal_position_sum" value is equal to self.target_length
						tf.equal(check_equal_position_sum, self.target_length), 
						tf.float32
					) # [N] 
			self.correct_count = tf.reduce_sum(correct_check) # scalar


		with tf.name_scope("saver"):
			self.saver = tf.train.Saver(max_to_keep=10000)
		
		sess.run(tf.global_variables_initializer())



	def encoder(self):
		# embedding lookup and scale
		encoder_input = tf.nn.embedding_lookup(
					self.embedding_table,
					self.sentence
				) # [N, self.sentence_length, self.embedding_size]
		if self.embedding_scale is True:
			encoder_input /= self.embedding_size**0.5

		encoder_input += self.PE[:self.sentence_length, :] 

		#drop out 
		encoder_input = tf.nn.dropout(encoder_input, keep_prob=self.keep_prob)
		
		mask = tf.expand_dims(self.sentence_mask, axis=-1) # [N, self.sentence_length, 1]
		encoder_input = encoder_input * mask # except padding

		#embedding = tf.nn.dropout(embedding, keep_prob=self.keep_prob)

		# stack encoder layer
		for i in range(1): #6
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
			encoder_input = Dense_add_norm

		return Dense_add_norm # [N, self.sentence_length, self.embedding_size]



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
		if self.embedding_scale is True:
			decoder_input /= self.embedding_size**0.5

		decoder_input += self.PE[:self.target_length, :]
		
		#drop out 
		decoder_input = tf.nn.dropout(decoder_input, keep_prob=self.keep_prob)

		# decoding
		decoder_output = self.decoder(
					decoder_input,
					encoder_embedding
				) # [N, self.target_length, self.voca_size]

		return decoder_output
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
		

	def decoder(self, decoder_input, encoder_embedding):
		# decoder_input: [N, self.target_length, self.embedding_size]
		# encoder_embedding: [N, self.sentence_length, self.embedding_size]
	
		# stack decoder layer
		for i in range(1):
			# Masked Multi-Head Attention
			Masked_Multihead_add_norm = self.multi_head_attention_add_norm(
						query=decoder_input, 
						key_value=decoder_input,
						mask=True,
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

		return linear # softmax는 cost 구할 때 seq2seq.sequence_loss에서 계산되므로 안함.



	def dense_add_norm(self, embedding, units, activation, name=None):
		# FFN(x) = max(0, x*W1+b1)*W2 + b2
		#변수공유  
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			# FFN
			inner_layer = tf.layers.dense(
					embedding, 
					units=2048, 
					activation=activation # relu
				) # [N, self.target_length, units]
			dense = tf.layers.dense(
					inner_layer, 
					units=units, 
					activation=None
				) # [N, self.target_length, self.embedding_size]
			
			#drop out 
			dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)

			# Add
			dense += embedding			
			# Layer Norm
			dense = tf.contrib.layers.layer_norm(dense,	begin_norm_axis=2)
	
		return dense 


	def multi_head_attention_add_norm(self, query, key_value, mask=False, activation=None, name=None):
		#변수공유
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
			V = tf.concat(
					#[N, key_value_sequence_length, self.embedding_size/8]이 8개 존재 
					tf.split(value=V, num_or_size_splits=8, axis=-1), 
					axis=0 
				) # [8*N, key_value_sequence_length, self.embedding_size/8]
			K = tf.concat(
					tf.split(value=K, num_or_size_splits=8, axis=-1), 
					axis=0
				) # [8*N, key_value_sequence_length, self.embedding_size/8]
			Q = tf.concat(
					tf.split(value=Q, num_or_size_splits=8, axis=-1), 
					axis=0
				) # [8*N, query_sequence_length, self.embedding_size/8]
			
			# Q * (K.T) and scaling ,  [8*N, query_sequence_length, key_value_sequence_length]
			score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size/8.0) 
		
			# masking
			if mask is True:
				score = score * self.decoder_mask # zero mask
				score = score + ((self.decoder_mask-1) * 1e+10) # -inf mask
				
			softmax = tf.nn.softmax(score, dim=2) # [8*N, query_sequence_length, key_value_sequence_length]
			attention = tf.matmul(softmax, V) # [8*N, query_sequence_length, self.embedding_size/8]			
			concat = tf.concat(
			 		# [N, query_sequence_length, self.embedding_size/8]이 8개 존재
					tf.split(value=attention, num_or_size_splits=8, axis=0),
					axis=-1
				) # [N, query_sequence_length, self.embedding_size]

			# Linear
			Multihead = tf.layers.dense(
					concat, 
					units=self.embedding_size, 
					activation=activation
				) # [N, query_sequence_length, self.embedding_size]

			#drop out 
			Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)

			# Add
			Multihead += query
			# Layer Norm			
			Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2) # [N, query_sequence_length, self.embedding_size]

			return Multihead


	
	def positional_encoding(self):
		alpha = 20
		PE = np.zeros([self.target_length + alpha, self.embedding_size])
		for pos in range(self.target_length + alpha): #충분히 크게 만들어두고 slice 해서 쓰자.
			for i in range(self.embedding_size//2): 
				PE[pos, 2*i] = np.sin( pos / np.power(10000, 2*i/self.embedding_size) )
				PE[pos, 2*i+1] = np.cos( pos / np.power(10000, 2*i/self.embedding_size) )
		
		return PE #[self.sentence_length, self.embedding_siz]
	

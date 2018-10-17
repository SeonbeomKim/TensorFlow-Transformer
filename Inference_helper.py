#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization

import tensorflow as tf
import numpy as np




class greedy_decoder:		
	def decode(self, decoder_fn, encoder_embedding, target_length, output_embedding_table, PE, go_idx):
		# encoder_embedding: [N, self.sentence_length, self.embedding_size]
		
		# decoder input preprocessing
		N = tf.shape(encoder_embedding)[0] # batchsize
		go_input = tf.one_hot(
					tf.zeros([N], tf.int32), 
					target_length, 
					on_value=go_idx, 
					off_value=-1
				) # [N, self.target_length]
		decoder_input = tf.nn.embedding_lookup(
					output_embedding_table, 
					go_input
				) # [N, self.target_length, self.embedding_size]
		decoder_input += PE[:target_length, :] 
		

		# greedy decoding
		decoder_output = []
		for index in range(target_length): 
			
			# get decoder_output of current postion
			current_output = decoder_fn(
						decoder_input, #decoder_input은 index마다 업데이트됨. 
						encoder_embedding, 
					) # [N, self.target_length, self.voca_size]

			# get current output and argmax
			current_output = current_output[:, index, :] # [N, self.voca_size]
			expand_current_output = tf.expand_dims(current_output, axis=1) # [N, 1, self.voca_size]	
			argmax_current_output = tf.argmax(expand_current_output, axis=-1) # [N, 1]

			# store embedding
			decoder_output.append(expand_current_output) 

			#Assign argmax_current_output(current position) to decoder_input(next position)
			if index < target_length-1:	
				pad_argmax_current_output = tf.pad( 
							argmax_current_output, 
							[[0,0], [index+1, target_length-index-2]], 
							mode='CONSTANT', 
							constant_values=-1
						) # [N, target_length]
				
				embedding_pad_argmax_current_output = tf.nn.embedding_lookup(
							output_embedding_table, 
							pad_argmax_current_output
						) # [N, target_length, self.embedding_size]
				
				decoder_input += embedding_pad_argmax_current_output # [N, target_length, self.embedding_size]

		# concat all position of decoder_output
		decoder_output = tf.concat(decoder_output, axis=1) # [N, target_length, self.voca_size]
		greedy_result = tf.argmax(decoder_output, axis=-1, output_type=tf.int32)
		
		return greedy_result, decoder_output
		'''
		go_input:
			go_idx -1 -1 -1 -1 -1 -1  # -1은 embedding_lookup하면 0처리되므로.
			go_idx -1 -1 -1 -1 -1 -1
			go_idx -1 -1 -1 -1 -1 -1
		-------------------------------------------------------------
		initial decoder_input:
			go_embedding+PE  go_embedding+PE  go_embedding+PE
			0+PE             0+PE             0+PE
			0+PE             0+PE             0+PE

			go_embedding+PE  go_embedding+PE  go_embedding+PE
			0+PE             0+PE             0+PE
			0+PE             0+PE             0+PE
		-------------------------------------------------------------			
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


class beam_decoder:
	def __init__(self, beam_size):
		self.beam_size = beam_size
		
	def decode(self, decoder_fn, encoder_embedding, target_length, output_embedding_table, PE, go_idx):
		# encoder_embedding: [N, self.sentence_length, self.embedding_size]
	
		# decoder input preprocessing
		N = tf.shape(encoder_embedding)[0] # batchsize
		go_input = tf.one_hot(
					tf.zeros([N], tf.int32), 
					target_length, 
					on_value=go_idx, 
					off_value=-1
				) # [N, target_length]
		decoder_input = tf.nn.embedding_lookup(
					output_embedding_table, 
					go_input
				) # [N, target_length, self.embedding_size]
		decoder_input += PE[:target_length, :] 

		# tile decoder_input  
		# https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/tile_batch
		decoder_input = tf.contrib.seq2seq.tile_batch(
					decoder_input, 
					self.beam_size
				) # [N*self.beam_size, target_length, self.embedding_size]

		encoder_embedding = tf.contrib.seq2seq.tile_batch(
					encoder_embedding,
					self.beam_size
				) # [N*self.beam_size, self.sentence_length, self.embedding_size]

		
		# beam_search decoding
		previous_top_k_prob = None
		previous_top_k_indices = None
		decoder_output = [] # [N*self.beam_size, target_length, self.voca_size]
		for index in range(target_length): 
			
			# get decoder_output of current postion
			current_output = decoder_fn(
						decoder_input, #decoder_input은 index마다 업데이트됨. 
						encoder_embedding, 
					) # [N*self.beam_size, target_length, self.voca_size]

			# get current output and softmax
			current_output = current_output[:, index, :] # [N*self.beam_size, self.voca_size]
			current_output_prob = tf.nn.softmax(current_output, dim=-1) # [N*self.beam_size, self.voca_size]
			
			# store embedding 
			expand_current_output = tf.expand_dims(current_output, axis=1) # [N*self.beam_size, 1, self.voca_size]
			decoder_output.append(expand_current_output) 

			# get top_k(beam_size) argmax prob, indices
			current_output_top_k_prob, current_output_top_k_indices = tf.nn.top_k(
						current_output_prob,
						self.beam_size
					) # [N*self.beam_size, beam_size], [N*self.beam_size, beam_size]


			if index == 0:
				beam_extract_index = tf.cast(tf.eye(self.beam_size), dtype=tf.bool) # [self.beam_size, self.beam_size]
				beam_extract_index = tf.tile(beam_extract_index, [N, 1]) # [N*self.beam_size, self.beam_size],  ([N, 1]: row*N, column*1)

				current_output_top_k_prob = tf.boolean_mask(current_output_top_k_prob, beam_extract_index) # [N*self.beam_size]
				# use log addition instead of multiplying(because of vanishing value)
				current_output_top_k_prob = tf.log(tf.reshape(current_output_top_k_prob, [-1, 1])) # [N*self.beam_size, 1] 
				previous_top_k_prob = current_output_top_k_prob
				
				current_output_top_k_indices = tf.boolean_mask(current_output_top_k_indices, beam_extract_index) # [N*self.beam_size]
				current_top_k_beam_output = tf.reshape(current_output_top_k_indices, [-1, 1]) # [N*self.beam_size, 1]
				previous_top_k_indices = current_top_k_beam_output
				
			else:
				current_output_top_k_prob = tf.log(tf.reshape(current_output_top_k_prob, [-1, self.beam_size])) # [N*self.beam_size, self.beam_size]

				# calc possible all prob and current maximum likelihood
				# previous_top_k_prob: [N*self.beam_size, 1]  but can addition(broadcasting)
				possible_all_prob = previous_top_k_prob + current_output_top_k_prob # [N*self.beam_size, self.beam_size] 누적 확률 계산
				possible_all_prob = tf.reshape(possible_all_prob, [-1, self.beam_size*self.beam_size]) # [N, self.beam_size*self.beam_size]
								
				# calc top_k argmax info about reshape_possible_all_prob
				possible_all_prob_top_k_prob, possible_all_prob_top_k_indices = tf.nn.top_k( # 누적 확률중 상위 beam_size개 추출
							possible_all_prob,
							self.beam_size
						) # [N, beam_size], [N, beam_size]
				
				# current maximum likelihood
				previous_top_k_prob = tf.reshape(possible_all_prob_top_k_prob, [-1, 1]) # [N*beam_size, 1]
				


				# calc possible all indices and top-k confidence beam indices
				tile_previous_top_k_indices = tf.contrib.seq2seq.tile_batch( 
							previous_top_k_indices, # [N*self.beam_size, index] 
							self.beam_size
						) # [N*self.beam_size*beam_size, index]
				current_output_top_k_indices = tf.reshape(current_output_top_k_indices, [-1, 1]) # [N*self.beam_size*beam_size, 1]
				
				possible_all_indices = tf.concat(
							(tile_previous_top_k_indices, current_output_top_k_indices), 
							axis=-1 
						) # [N*self.beam_size*beam_size, index+1] 가능한 모든 path 계산

				possible_all_indices = tf.reshape(
							possible_all_indices, 
							[N, self.beam_size*self.beam_size, -1]
						) # [N, self.beam_size*beam_size, index+1] split batch
				
				#https://stackoverflow.com/questions/36088277/how-to-select-rows-from-a-3-d-tensor-in-tensorflow
				possible_all_prob_top_k_indices = tf.reshape(possible_all_prob_top_k_indices, [-1]) # [N*self.beam_size]
				#https://www.tensorflow.org/api_docs/python/tf/stack
				gather_indices = tf.stack(
							[tf.range(tf.shape(possible_all_prob_top_k_indices)[0])//self.beam_size, 
								possible_all_prob_top_k_indices], 
							axis=-1
						) # [N*self.beam_size, 2]
				gather_indices = tf.reshape(gather_indices, [-1, self.beam_size, 2]) # [N, self.beam_size, 2]
			
				# gather top-k confidence beam indices
				top_k_confidence_beam_indices = tf.gather_nd(
							possible_all_indices, 
							gather_indices
						) # [N, self.beam_size, index+1]
				previous_top_k_indices = tf.reshape(top_k_confidence_beam_indices, [N*self.beam_size, -1]) # [N*self.beam_size, index+1]
				
				current_top_k_beam_output = tf.gather_nd(
							possible_all_indices[:, :, -1:], 
							gather_indices
						) # [N, self.beam_size, 1]						
				current_top_k_beam_output = tf.reshape(current_top_k_beam_output, [-1, 1]) # [N*self.beam_size, 1]
				

			if index < target_length-1:	
				pad_top_k_indices = tf.pad( 
							current_top_k_beam_output, 
							[[0,0], [index+1, target_length-index-2]], 
							mode='CONSTANT', 
							constant_values=-1
						) # [N*beam_size, target_length]
				
				embedding_pad_top_k_indices = tf.nn.embedding_lookup(
							output_embedding_table, 
							pad_top_k_indices
						) # [N*beam_size, target_length, self.embedding_size]		
			
				decoder_input += embedding_pad_top_k_indices


		# previous_top_k_prob to multiply value
		# log(a*b*c) == log(a)+log(b)+log(c) == k
		# e^k == a*b*c
		# a*b*c == tf.exp(previous_top_k_prob)
		# previous_top_k_prob = tf.exp(previous_top_k_prob)

		previous_top_k_indices = tf.reshape(
					previous_top_k_indices, # [N*self.beam_size, target_length]
					[-1, self.beam_size, target_length]
				) # [N, self.beam_size, target_length]
		beam_result = previous_top_k_indices[:, 0, :] # [N, target_length]  #dtype=tf.int32

		decoder_output = tf.concat(decoder_output, axis=1) # [N*self.beam_size, target_length, self.voca_size]
		decoder_output = tf.reshape(
					decoder_output,
					[N, self.beam_size, target_length, -1]
				) # [N, self.beam_size, target_length, self.voca_size]
		decoder_output = decoder_output[:, 0, :, :] # [N, target_length, self.voca_size]

		return beam_result, decoder_output






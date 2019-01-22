import tensorflow as tf
import numpy as np
import nltk

class greedy:
	def __init__(self, sess, model, go_idx):
		self.sess = sess
		self.model = model
		self.go_idx = go_idx

	def decode(self, encoder_input, target_length):
		sess = self.sess
		model = self.model
		encoder_input = np.array(encoder_input, dtype=np.int32)
		
		input_token = np.zeros([encoder_input.shape[0], target_length+1], np.int32) # go || target_length
		input_token[:, 0] = self.go_idx

		encoder_embedding = sess.run(model.encoder_embedding,
				{
					model.encoder_input:encoder_input, 
					model.keep_prob:1
				}
			) # [N, self.encoder_input_length, self.embedding_size]
		
		for index in range(target_length):
			current_pred = sess.run(model.decoder_pred,
					{
						model.encoder_embedding:encoder_embedding,
						model.decoder_input:input_token[:, :index+1],
						model.keep_prob:1
					}
				) # [N, target_length+1]
			input_token[:, index+1] = current_pred[:, index]

		# [N, target_length]
		return input_token[:, 1:]




class beam:
	def __init__(self, sess, model, go_idx, eos_idx, beam_width, length_penalty=0.6):
		self.sess = sess
		self.model = model
		self.go_idx = go_idx
		self.eos_idx = eos_idx
		self.beam_width = beam_width
		self.length_penalty = length_penalty
		self.build_beam_graph()
	

	def build_beam_graph(self):
		model = self.model

		self.time_step = tf.placeholder(tf.int32, name='time_step_placeholder')

		self.tile_encoder_embedding = tf.contrib.seq2seq.tile_batch(model.encoder_embedding, self.beam_width)
		tile_current_embedding = model.decoder_embedding[:, self.time_step, :] # [N*beam_width, voca_size]

		top_k_prob, top_k_indices = tf.nn.top_k(
				tf.nn.softmax(tile_current_embedding, dim=-1), # [N*beam_width, self.voca_size]
				self.beam_width
			) # [N*beam_width, beam_width], [N*beam_width, beam_width]

		# lp(length_penalty) https://arxiv.org/pdf/1609.08144.pdf
		Y_length = tf.to_float(self.time_step) + 1
		lp = ((5. + Y_length)**self.length_penalty) / ((5. + 1.)**self.length_penalty)
		self.top_k_prob = tf.log(tf.reshape(top_k_prob, [-1, 1])) / lp # [N*beam_width*beam_width, 1]
		self.top_k_indices = tf.reshape(top_k_indices, [-1, 1]) # [N*beam_width*beam_width, 1]


	def decode(self, encoder_input, target_length):	
		sess = self.sess
		model = self.model
		beam_width = self.beam_width

		encoder_input = np.array(encoder_input, dtype=np.int32)
	
		N = encoder_input.shape[0]
		for_indexing = np.arange(N).reshape(-1, 1) * beam_width * beam_width # [N, 1]
		
		# for eos check,  one-initialize
		is_previous_eos = np.ones([N*beam_width*beam_width, 1], dtype=np.float32) 

		input_token = np.zeros([N*beam_width, target_length+1], np.int32) # go || target_length
		input_token[:, 0] = self.go_idx
	
		encoder_embedding = sess.run(self.tile_encoder_embedding, 
				{
					model.encoder_input:encoder_input, 
					model.keep_prob:1,
				}
			) # [N*beam_width, self.encoder_input_length, self.embedding_size]
		
		for index in range(target_length):
			prob, indices = sess.run([self.top_k_prob, self.top_k_indices],
					{
						model.encoder_embedding:encoder_embedding,
						model.decoder_input:input_token[:, :index+1],
						model.keep_prob:1,
						self.time_step:index,
					}
				) # each [N*beam_width*beam_width, 1]

			if index == 0:
				prob = prob.reshape([-1, beam_width, beam_width]) # [N, beam_width, beam_width]
				prob = prob.transpose([0, 2, 1]) # [N, beam_width, beam_width]
				prob = prob.reshape([-1, 1]) # [N*beam_width*beam_width, 1]
				indices = indices.reshape([-1, beam_width, beam_width]) # # [N, beam_width, beam_width]
				indices = indices.transpose([0, 2, 1]) # [N, beam_width, beam_width]
				indices = indices.reshape([-1, 1]) # [N*beam_width*beam_width, 1]				
				input_token[:, 1] = indices[np.arange(0, N*beam_width*beam_width, beam_width)].reshape(-1) # [N*beam_width]
				# save
				prob_list = prob # [N*beam_width*beam_width, 1]
				indices_list = indices # [N*beam_width*beam_width, 1]				

			else:
				# 이전 output 중에 한번이라도 eos가 있으면 prob 반영 안함. eos가 없으면 1, 있으면 0
				is_previous_eos *= (indices_list[:, -1:] != self.eos_idx) # [N*beam_width*beam_width, 1]
				masked_prob = prob * is_previous_eos # [N*beam_width*beam_width, 1]	
				prob_list += masked_prob # [N*beam_width*beam_width, 1]
				indices_list = np.concatenate((indices_list, indices), axis=1) # [N*beam_width*beam_width, index+1]
			
				batch_split_prob_list = prob_list.reshape([-1, beam_width*beam_width]) # [N, beam_width*beam_width]
				top_k_indices = np.argsort(-batch_split_prob_list)[:, :beam_width] # -붙여야 내림차순 정렬. [N, beam_width]
				top_k_indices += for_indexing # [N, beam_width]
				top_k_indices = top_k_indices.reshape(-1) # [N*beam_width]
				
				is_previous_eos = is_previous_eos[top_k_indices] # [N*beam_width, 1]
				top_k_prob = prob_list[top_k_indices] # [N*beam_width, 1] 
				indices_list = indices_list[top_k_indices] # [N*beam_width, index+1]
				input_token[:, 1:index+2] = indices_list

				if index < target_length-1:
					# save
					is_previous_eos = np.tile(is_previous_eos, beam_width) # [N*beam_width, beam_width]
					is_previous_eos = is_previous_eos.reshape(N*beam_width*beam_width, 1) # [N*beam_width*beam_width, 1]
					indices_list = np.tile(indices_list, beam_width) # [N*beam_width, beam_width*(index+1)]
					indices_list = indices_list.reshape(N*beam_width*beam_width, -1) # [N*beam_width*beam_width, (index+1)]
					prob_list = np.tile(top_k_prob, beam_width) # [N*beam_width, beam_width]
					prob_list = prob_list.reshape(N*beam_width*beam_width, 1) # [N*beam_width*beam_width, 1]
		
		indices_list = indices_list.reshape(N, beam_width, target_length)

		# [N, target_length]
		return indices_list[:, 0, :] # batch마다 가장 probability가 높은 결과 리턴.




class utils:
	def __init__(self):
		pass

	def bleu(self, target, pred):
		smoothing = nltk.translate.bleu_score.SmoothingFunction()
		score = nltk.translate.bleu_score.corpus_bleu(target, pred, smoothing_function=smoothing.method0)
		#score = nltk.translate.bleu_score.corpus_bleu(target, pred, smoothing_function=smoothing.method4)
		return score



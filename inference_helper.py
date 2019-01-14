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
		go_idx = self.go_idx
		encoder_input = np.array(encoder_input, dtype=np.int32)
		
		input_token = np.zeros([encoder_input.shape[0], target_length+1], np.int32) # go || target_length
		input_token[:, 0] = go_idx
		#decoder_embedding = []

		# 이렇게 안하면 decoder time step 계산할 때마다 encoder embedding 재계산해야해서 느림.
		encoder_embedding, encoder_input_mask = sess.run([model.encoder_embedding,  model.encoder_input_mask],
				{
					model.encoder_input:encoder_input, 
					model.keep_prob:1
				}
			) # [N, self.encoder_input_length, self.embedding_size]
		
		for index in range(target_length):
			#current_pred, current_embedding = sess.run([model.infer_pred, model.infer_embedding],
			current_pred = sess.run(model.decoder_pred,
					{
						model.encoder_input_mask:encoder_input_mask,
						model.encoder_embedding:encoder_embedding,
						model.decoder_input:input_token[:, :-1],
						model.keep_prob:1
					}
				) # [N, target_length+1]
			input_token[:, index+1] = current_pred[:, index]
			#decoder_embedding.append(current_embedding[:, index:index+1, :]) # [N, 1, self.voca_size]

		# [N, target_length]
		return input_token[:, 1:]#, np.concatenate(decoder_embedding, axis=1) #except 'go'


'''
class beam:
	def __init__(self, sess, model, go_idx, beam_width=2):
		self.sess = sess
		self.model = model
		self.go_idx = go_idx
		self.beam_width = beam_width

	def decode(self, encoder_input, target_length):	
		sess = self.sess
		model = self.model
		go_idx = self.go_idx
		beam_width = self.beam_width
		encoder_input = np.array(encoder_input, dtype=np.int32)
	
		N = encoder_input.shape[0]
		for_indexing = np.arange(N).reshape(-1, 1) * beam_width * beam_width # [N, 1]

		input_token = np.zeros([N*beam_width, target_length+1], np.int32) # go || target_length
		input_token[:, 0] = go_idx
		
		encoder_embedding, encoder_input_mask = sess.run([tf.contrib.seq2seq.tile_batch(model.encoder_embedding, beam_width),  tf.contrib.seq2seq.tile_batch(model.encoder_input_mask, beam_width)],
				{
					model.encoder_input:encoder_input, 
					model.keep_prob:1, 
				}
			) # [N*beam_width, self.encoder_input_length, self.embedding_size]
	

		for index in range(target_length):
			prob, indices = sess.run([model.top_k_prob, model.top_k_indices], 
					{
						model.encoder_input_mask:encoder_input_mask,
						model.feed_encoder_embedding:encoder_embedding, 
						model.decoder_input:input_token[:, :-1], 
						model.keep_prob:1,
						model.time_step:index,
						model.beam_width:beam_width
					} # prob is log value
				) # [N*beam_width*beam_width, 1], # [N*beam_width*beam_width, 1]
			
			if index == 0:
				prob = prob.reshape([-1, beam_width, beam_width]) # [N, beam_width, beam_width]
				indices = indices.reshape([-1, beam_width, beam_width]) # # [N, beam_width, beam_width]

				prob = prob.transpose([0, 2, 1]) # [N, beam_width, beam_width]
				indices = indices.transpose([0, 2, 1]) # [N, beam_width, beam_width]
				
				prob = prob.reshape([-1, 1]) # [N*beam_width*beam_width, 1]
				indices = indices.reshape([-1, 1]) # [N*beam_width*beam_width, 1]

				prob_list = prob
				indices_list = indices
				
				input_token[:, index+1] = indices[np.arange(0, N*beam_width*beam_width, beam_width)].reshape(-1)
				

			else:
				indices_list = np.concatenate((indices_list, indices), axis=1) # [N*beam_width*beam_width, index+1]
				beam_score = prob_list[-1] + prob # [N*beam_width*beam_width, 1]
			
				batch_split_beam_score = beam_score.reshape([-1, beam_width*beam_width]) # [N, beam_width*beam_width]
				
				# tensor쓰면 느림 
				# [N, beam_width], [N, beam_width]
				#top_k_prob, top_k_indices = sess.run(tf.nn.top_k(batch_split_beam_score, beam_width))
				#top_k_prob = top_k_prob.reshape(-1, 1) # [N*beam_width, 1]
				
				# 정렬안하고 top_k 개 효율적으로 뽑는것 찾기. O(n) 인 np.argpartition는 top_k를 뽑긴 하지만 순서가 없음.
				# 정렬하는경우 time: N * (beam_width*beam_width)log(beam_width*beam_width)
				# 정렬안하고 힙으로 뽑는다면, buildheap: N * (beam_width*beam_width), max_extract: N, heapify: N*log(beam_width*beam_width)
				#	so, beam_width번 뽑는 time:  N * (beam_width*beam_width) + N*beam_width + N*beam_width*log(beam_width*beam_width)
				# 		=>  N * (beam_width*beam_width).
				# 즉 정렬하는경우 log(beam_width*beam_width) 만큼 더 걸리는데 beam_width는 어쩌피 작은값 쓰니까. 정렬해도됨.

				top_k_indices = np.argsort(-batch_split_beam_score)[:, :beam_width] # -붙여야 내림차순 정렬. [N, beam_width]
				top_k_prob = beam_score[top_k_indices]
				top_k_prob = top_k_prob.reshape(-1, 1) # [N*beam_width, 1]
				
				top_k_indices += for_indexing # [N, beam_width]
				top_k_indices = top_k_indices.reshape([-1]) # [N*beam_width]

				indices_list = indices_list[top_k_indices] # [N*beam_width, index+1]
				input_token[:, 1:index+2] = indices_list

		
				if index < target_length-1:
					# [N*beam_width*beam_width, index+1], [N*beam_width*beam_width, 1]
					# 느림 
					#indices_list, prob_list = sess.run([tf.contrib.seq2seq.tile_batch(indices_list, beam_width), tf.contrib.seq2seq.tile_batch(top_k_prob, beam_width)])
					
					indices_list = np.tile(indices_list.reshape(N*beam_width, 1, -1), beam_width) # [N*beam_width*beam_width, 1, index+1]
					prob_list = np.tile(top_k_prob.reshape(N*beam_width, 1, -1), beam_width) # [N*beam_width*beam_width, 1, 1]

					indices_list = indices_list.reshape(N*beam_width*beam_width, -1)
					prob_list = prob_list.reshape(N*beam_width*beam_width, -1)
		
		indices_list = indices_list.reshape(N, beam_width, target_length)

		# [N, target_length]
		return indices_list[:, 0, :] # batch마다 가장 probability가 높은 결과 리턴.
'''



class utils:
	def __init__(self):
		pass

	def bleu(self, target, pred):
		smoothing = nltk.translate.bleu_score.SmoothingFunction()
		score = nltk.translate.bleu_score.corpus_bleu(target, pred, smoothing_function=smoothing.method0)
		#score = nltk.translate.bleu_score.corpus_bleu(target, pred, smoothing_function=smoothing.method4)
		return score



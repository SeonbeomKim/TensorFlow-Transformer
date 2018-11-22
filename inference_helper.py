import tensorflow as tf
import numpy as np
import transformer

def greedy(sentence, model, go_idx, target_length):
	input_token = np.zeros([sentence.shape[0], target_length+1], np.int32) # go || target_length
	input_token[:, 0] = go_idx
	#decoder_embedding = []

	# 이렇게 안하면 decoder time step 계산할 때마다 encoder embedding 재계산해야해서 느림.
	encoder_embedding = sess.run(model.encoder_embedding,  
				{
					model.sentence:sentence, 
					model.keep_prob:1
				}
			) # [N, self.sentence_length, self.embedding_size]
	
	for index in range(target_length):
		current_pred, current_embedding = sess.run([model.infer_pred, model.infer_embedding],
					{
						model.feed_encoder_embedding:encoder_embedding,
						model.target:input_token,
						model.keep_prob:1
					}
				) # [N, target_length+1]
		input_token[:, index+1] = current_pred[:, index]
		#decoder_embedding.append(current_embedding[:, index:index+1, :]) # [N, 1, self.voca_size]

	return input_token[:, 1:]#, np.concatenate(decoder_embedding, axis=1) #except 'go'



def beam(sentence, model, go_idx, target_length, beam_width):
	N = sentence.shape[0]
	for_indexing = np.arange(N).reshape(-1, 1) * beam_width * beam_width # [N, 1]

	input_token = np.zeros([N*beam_width, target_length+1], np.int32) # go || target_length
	input_token[:, 0] = go_idx
	
	encoder_embedding = sess.run(tf.contrib.seq2seq.tile_batch(model.encoder_embedding, beam_width),  
				{
					model.sentence:sentence, 
					model.keep_prob:1, 
				}
			) # [N*beam_width, self.sentence_length, self.embedding_size]
	

	for index in range(target_length):
		prob, indices = sess.run([model.top_k_prob, model.top_k_indices], 
					{
						model.feed_encoder_embedding:encoder_embedding, 
						model.target:input_token, 
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
	return indices_list[:, 0, :] # batch마다 가장 probability가 높은 결과 리턴.


def testcode():
	sentence = np.array([[1,2,0],[4,0,0]])
	tar = [[1,2,3],[4,5,6]]
	zz = [[0, 1, 2], [0, 4, 5]]

	go_idx = 1

	import time


	start = time.time()
	print(beam(sentence, model, go_idx, target_length=5, beam_width=1))
	print('beam', time.time()-start)

	start = time.time()
	print(beam(sentence, model, go_idx, target_length=5, beam_width=2))
	print('beam', time.time()-start)

	start = time.time()
	print(beam(sentence, model, go_idx, target_length=5, beam_width=3))
	print('beam', time.time()-start)

	print()
	print()
	print()
	print()



	start = time.time()
	print(greedy(sentence, model, go_idx, target_length=5))
	print('greedy', time.time()-start)

	start = time.time()
	print(greedy(sentence, model, go_idx, target_length=5))
	print('greedy', time.time()-start)

	start = time.time()
	print(greedy(sentence, model, go_idx, target_length=5))
	print('greedy', time.time()-start)



sess = tf.Session()

model = transformer.Transformer(
			sess = sess,
			voca_size = 10, 
			embedding_size = 8, 
			is_embedding_scale = True, 
			PE_sequence_length = 300,
			encoder_decoder_stack = 1,
			go_idx=0, 
			eos_idx=1, 
			pad_idx=0,
			label_smoothing=0.1
		)


testcode()
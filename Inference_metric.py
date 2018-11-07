import tensorflow as tf
import numpy as np


class metric:
	def __init__(self, target_length, eos_idx, pad_idx):
		self.target_length = target_length
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx


	#inference cost
	def cost(self, smoothing_target_one_hot, infer_embedding, target_pad_mask):
		# calc infer_cost
		infer_cost = tf.nn.softmax_cross_entropy_with_logits(
					labels = smoothing_target_one_hot, 
					logits = infer_embedding
				) # [N, self.target_length]
		infer_cost = infer_cost * target_pad_mask # except pad
		infer_cost = tf.reduce_sum(infer_cost) / tf.reduce_sum(target_pad_mask)
		return infer_cost



	#inference accuracy
	def accuracy(self, target, infer_pred_except_eos):
		target = tf.convert_to_tensor(target, dtype=tf.int32)
		
		target_eos_mask = tf.cast( #sequence_mask처럼 생성됨.
				tf.not_equal(target, self.pad_idx) & tf.not_equal(target, self.eos_idx),
				dtype=tf.int32
			) # [N, target_length] (include eos)

		target_except_eos = target * target_eos_mask # masking pad
		target_except_eos += (target_eos_mask - 1) # the value of the masked position is -1
		
		# correct check
		check_equal_position = tf.cast(
					tf.equal(target_except_eos, infer_pred_except_eos), 
					dtype=tf.int32
				) # [N, self.target_length]
	
		check_equal_position_sum = tf.reduce_sum( 	#if use mean, 0.9999999 is equal to 1, so use sum.
					check_equal_position, 
					axis=-1
				) # [N]
		
		correct_check = tf.cast( #if correct: "check_equal_position_sum" value is equal to self.target_length
					tf.equal(check_equal_position_sum, self.target_length), 
					tf.float32
				) # [N] 
		correct_count = tf.reduce_sum(correct_check) # scalar
		return correct_count
	

	#inference bleu
	def bleu(self):
		pass
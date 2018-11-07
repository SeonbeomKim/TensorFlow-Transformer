import tensorflow as tf
import numpy as np


class utils:
	def __init__(self, model, go_idx, eos_idx, pad_idx):
		self.model = model
		self.go_idx = go_idx
		self.eos_idx = eos_idx
		self.pad_idx = pad_idx


	def inference(self, decode_helper, is_embedding_scale, target_length):
		model = self.model

		infer_pred, infer_embedding = decode_helper.decode(
					decoder_fn=model.decoder, 
					encoder_embedding=model.encoder_embedding, 
					target_length=target_length, 
					PE=model.PE, 
					go_idx=self.go_idx, 
					embedding_table=model.embedding_table, 
					is_embedding_scale=is_embedding_scale
				)	
		# inference_output masking(remove eos, pad)
		infer_first_eos = tf.argmax(
					tf.cast( tf.equal(infer_pred, self.eos_idx), tf.int32), 
					axis=-1
				) # [N]
		infer_eos_mask = tf.sequence_mask(
					infer_first_eos,
					maxlen=target_length,
					dtype=tf.int32
				)
		infer_pred_except_eos = infer_pred * infer_eos_mask
		infer_pred_except_eos += (infer_eos_mask - 1) # the value of the masked position is -1
		return infer_pred_except_eos, infer_embedding



	#inference cost
	def cost(self, infer_embedding):
		model = self.model

		# calc infer_cost
		infer_cost = tf.nn.softmax_cross_entropy_with_logits(
					labels = model.smoothing_target_one_hot, 
					logits = infer_embedding
				) # [N, self.target_length]
		infer_cost = infer_cost * model.target_pad_mask # except pad
		infer_cost = tf.reduce_sum(infer_cost) / tf.reduce_sum(model.target_pad_mask)
		return infer_cost



	#inference accuracy
	def accuracy(self, target, infer_pred_except_eos, target_length):
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
					tf.equal(check_equal_position_sum, target_length), 
					tf.float32
				) # [N] 
		correct_count = tf.reduce_sum(correct_check) # scalar
		return correct_count
	

	#inference bleu
	def bleu(self):
		pass
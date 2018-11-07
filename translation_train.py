#https://arxiv.org/abs/1706.03762 Attention Is All You Need(Transformer)
#https://arxiv.org/abs/1607.06450 Layer Normalization

import Transformer
import Decode_helper # greedy, beam
import numpy as np
import tensorflow as tf # 1.4
import Inference_utils # metric, inference
import bucket_data_helper
import datetime
import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

train_set_path = './bpe_dataset/train_bucket_concat_dataset.npy'
valid_set_path = './bpe_dataset/valid_bucket_concat_dataset.npy'
bpe2idx_path = './bpe_dataset/bpe2idx.npy'

saver_path = './saver/'
tensorboard_path = './tensorboard/'


def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data


def train(model, data, lr):
	loss = 0
	total_data_len = 0

	data.shuffle()

	while True:
		batch = data.get_batch()
		if batch is 0:
			break # epoch end

		batch_data, bucket = batch[0], batch[1]
		total_data_len += len(batch_data)
	
		sentence = batch_data[:, :bucket[0]]
		target = batch_data[:, bucket[0]:]

		train_loss, _ = sess.run([model.train_cost, model.minimize], 
							{
								model.lr:lr,
								model.sentence:sentence, 
								model.target:target, 
								model.keep_prob:0.9 # dropout rate = 0.1		
							}
						)
		loss += train_loss

	return loss/total_data_len


def validation(model, data, decode_helper, infer_utils):
	loss = 0
	total_data_len = 0
	
	while True:
		batch = data.get_batch()
		if batch is 0:
			break # epoch end

		batch_data, bucket = batch[0], batch[1]
		total_data_len += len(batch_data)
	
		sentence = batch_data[:, :bucket[0]]
		target = batch_data[:, bucket[0]:]
		
		infer_pred_except_eos, infer_embedding = infer_utils.inference(
					decode_helper=decode_helper, 
					is_embedding_scale=True, 
					target_length=bucket[1]
				)
		infer_cost = infer_utils.cost(infer_embedding)
		infer_accuracy = infer_utils.accuracy(target, infer_pred_except_eos, target_length=bucket[1])
		
		vali_loss = sess.run(infer_cost, 
					{
						model.sentence:sentence, 
						model.target:target, 
						model.keep_prob:1
					}
				)

		loss += vali_loss

	return loss/total_data_len




def run(model, sess, train_batch, valid_batch, decode_helper, infer_utils,  restore=0):
	with tf.name_scope("tensorboard"):
		train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')
		vali_loss_tensorboard = tf.placeholder(tf.float32, name='vali_loss')
		#test_accuracy_tensorboard = tf.placeholder(tf.float32, name='test')

		train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
		vali_summary = tf.summary.scalar("vali_loss", vali_loss_tensorboard)
		#test_summary = tf.summary.scalar("test_accuracy", test_accuracy_tensorboard)
				
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

	if not os.path.exists(saver_path):
		print("create save directory")
		os.makedirs(saver_path)


	if restore != 0:
		print("restore", restore)
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
		
		vali_loss = validation(model, valid_batch, decode_helper, infer_utils)
		print("epoch:", restore, "\tvali_loss:", vali_loss, '\ttime:', datetime.datetime.now())


	
	for epoch in range(restore+1, 20000+1):
		lr = (embedding_size**-0.5) * min( (epoch**-0.5), (epoch*(warmup_steps**-1.5)) )

		train_loss = train(model, train_batch, lr)
		print("epoch:", epoch, 'train_loss:', train_loss, 'time:', datetime.datetime.now(), 'lr:', lr)
		
		if epoch % 10 == 0:
			model.saver.save(sess, saver_path+str(epoch)+".ckpt")


		if epoch % test_epoch == 0:
			vali_loss = validation(model, valid_batch, decode_helper, infer_utils)
			print("epoch:", epoch, "train_loss:", train_loss, "\tvali_loss:", vali_loss, '\tlr:', lr, '\ttime:', datetime.datetime.now())
		
			#accuracy = test(model, testset)
			#print("epoch:", epoch, "train_loss:", train_loss, "\tvali_loss:", vali_loss, "\taccuracy:", accuracy, '\tlr:', lr, '\ttime:', time.time()-prev)

			# tensorboard
			summary = sess.run(merged, {
						train_loss_tensorboard:train_loss, 
						vali_loss_tensorboard:vali_loss,
						#test_accuracy_tensorboard:accuracy, 
					}
			)
			writer.add_summary(summary, epoch)




	
		

print('Data read')
train_set = load_dictionary(train_set_path)
train_batch = bucket_data_helper.bucket_data(train_set, iter=True, batch_token = 16000)

valid_set = load_dictionary(valid_set_path)
valid_batch = bucket_data_helper.bucket_data(valid_set, iter=True, batch_token = 16000)

bpe2idx = load_dictionary(bpe2idx_path)




warmup_steps = 1000
test_epoch = 50

print('Transformer read')
sess = tf.Session()
model = Transformer.Transformer(
			sess = sess,
			voca_size = len(bpe2idx), 
			embedding_size = 256, 
			is_embedding_scale = True, 
			PE_sequence_length = 200,
			encoder_decoder_stack = 1,
			go_idx=bpe2idx['</g>'], 
			eos_idx=bpe2idx['</e>'], 
			pad_idx=-1,
			label_smoothing=0.1
		)

decode_helper = Decode_helper.greedy_decoder()
infer_utils = Inference_utils.utils(model=model, go_idx=bpe2idx['</g>'], eos_idx=bpe2idx['</e>'], pad_idx=-1)
print(decode_helper)


print('run')
run(model, sess, train_batch, valid_batch, decode_helper, infer_utils, 90)

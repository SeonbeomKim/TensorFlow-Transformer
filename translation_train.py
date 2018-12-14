import tensorflow as tf
import numpy as np
import transformer
import inference_helper
import os
import bucket_data_helper
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

train_set_path = './bpe_dataset/train_bucket_concat_dataset.npy'
vali_set_path = './bpe_dataset/valid_bucket_concat_dataset.npy'
bpe2idx_path = './bpe_dataset/bpe2idx.npy'

#test_set_path = './calc/test_set.csv'
saver_path = './saver/'
tensorboard_path = './tensorboard/'


warmup_steps = 4000 #//10


def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data

def save_dictionary(path, dictionary):
	np.save(path, dictionary)


def train(model, data, lr):
	total_data_len = 0
	loss = 0

	data.shuffle()

	while True:
		batch = data.get_batch()
		if batch is 0:
			break # epoch end

		batch_data, encoder_input_length = batch[0], batch[1][0]
		
		encoder_input = batch_data[:, :encoder_input_length]
		decoder_input = batch_data[:, encoder_input_length:-1] 
		target = batch_data[:, encoder_input_length+1:] # except '</g>'


		if total_data_len == 0:
			train_loss, _, a, b = sess.run([model.train_cost, model.minimize, model.decoder_pred, model.target_pad_mask], 
			#train_loss, _, a, b = sess.run([model.train_cost, model.minimize, model.decoder_pred_except_eos, model.target_pad_mask], 
			#train_loss, _ = sess.run([model.train_cost, model.minimize], 
							{
								model.lr:lr,
								model.encoder_input:encoder_input, 
								model.decoder_input:decoder_input,
								model.target:target, 
								model.keep_prob:0.9 # dropout rate = 0.1		
							}
						)
			print("decode\n", a[:1])
			print("target\n", (target[:1]*b[:1]).astype(np.int32))

		else:	
			train_loss, _ = sess.run([model.train_cost, model.minimize], 
		#train_loss, _ = sess.run([model.train_cost, model.minimize], 
							{
								model.lr:lr,
								model.encoder_input:encoder_input, 
								model.decoder_input:decoder_input,
								model.target:target, 
								model.keep_prob:0.9 # dropout rate = 0.1		
							}
						)

		total_data_len += len(batch_data)
		loss += train_loss
	
	return loss/total_data_len



def validation(model, data):
	total_data_len = 0
	loss = 0

	while True:
		batch = data.get_batch()
		if batch is 0:
			break # epoch end

		batch_data, encoder_input_length = batch[0], batch[1][0]
		encoder_input = batch_data[:, :encoder_input_length]
		decoder_input = batch_data[:, encoder_input_length:-1] 
		target = batch_data[:, encoder_input_length+1:] # except '</g>'

		vali_loss = sess.run(model.train_cost, # greedy나 beamsearch embedding로 계산하는것이 엄밀함.
							{
								model.encoder_input:encoder_input, 
								model.decoder_input:decoder_input,
								model.target:target, 
								model.keep_prob:1
							}
						)

		total_data_len += len(batch_data)
		loss += vali_loss		


	return loss/total_data_len


"""
def test(model, data):
	batch_size = 256 #// 2
	correct = 0

	#for i in range( int(np.ceil(len(data)/batch_size)) ):
	for i in tqdm(range( int(np.ceil(len(data)/batch_size)) ), ncols=50):
		batch = data[batch_size * i: batch_size * (i + 1)]
		
		encoder_input = batch[:, :sentence_length]
		decoder_input = batch[:, sentence_length:-1]
		target = batch[:, sentence_length+1:]

		infer_pred = greedy.decode(encoder_input, target_length=target.shape[1])
		infer_pred_except_eos = sess.run(model.except_eos, {model.feed_for_except_eos:infer_pred})
		target_except_eos = sess.run(model.except_eos, {model.feed_for_except_eos:target})

		count = utils.correct(infer_pred_except_eos, target_except_eos)
		correct += count
		'''
		print('\ninfer_pred\n', infer_pred[:3], '\n')
		print('infer_pred_except_eos\n', infer_pred_except_eos[:3], '\n')
		print('target\n', target[:3], '\n')
		print('correct:', correct, '\n')
		'''
		print('correct:', correct, '\n')

		break
	return correct/len(data)
"""

def run(model, trainset, valiset, testset, restore=0):
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
	
	import datetime
	print('start', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

	for epoch in range(restore+1, 20000+1):
		lr = (embedding_size**-0.5) * min( (epoch**-0.5), (epoch*(warmup_steps**-1.5)) )
		#lr = 0.01

		train_loss = train(model, trainset, lr)
		print('train', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		vali_loss = validation(model, valiset)
		print('vali', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
		#correct = test(model, testset)
		#print('test', datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

		print("epoch:", epoch, 'train_loss:', train_loss,  'vali_loss:', vali_loss, 'lr:', lr, '\n')

		# tensorboard
		summary = sess.run(merged, {
					train_loss_tensorboard:train_loss, 
					vali_loss_tensorboard:vali_loss,
					#test_accuracy_tensorboard:accuracy, 
				}
		)
		writer.add_summary(summary, epoch)

		if (epoch) % 100 == 0:
			model.saver.save(sess, saver_path+str(epoch)+".ckpt")

		

		#accuracy = test(model, testset)






print('Data read')
train_set_bucket = load_dictionary(train_set_path)
train_set = bucket_data_helper.bucket_data(train_set_bucket, iter=True, batch_token = 12000) # batch_token // len(sentence||target token) == batch_size
vali_set_bucket = load_dictionary(vali_set_path)
vali_set = bucket_data_helper.bucket_data(vali_set_bucket, iter=True, batch_token = 12000) # batch_token // len(sentence||target token) == batch_size
bpe2idx = load_dictionary(bpe2idx_path)

embedding_size = 256

print("Model read")
sess = tf.Session()

model = transformer.Transformer(
			sess = sess,
			voca_size = len(bpe2idx), 
			embedding_size = embedding_size, 
			is_embedding_scale = True, 
			PE_sequence_length = 300,
			encoder_decoder_stack = 1*3,
			multihead_num = 8,
			go_idx=bpe2idx['</g>'], 
			eos_idx=bpe2idx['</e>'], 
			pad_idx=bpe2idx['</p>'],
			label_smoothing=0.1
		)
greedy = inference_helper.greedy(sess, model, bpe2idx['</g>'])
utils = inference_helper.utils()

print('run')
run(model, train_set, vali_set, None)

#print(sess.run(model.embedding_table))

#testcode()


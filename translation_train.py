import tensorflow as tf
import numpy as np
import transformer
import inference_helper
import bucket_data_helper
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#bucket = [(100, 130), (140, 170), (180, 210)]
bucket = [(10, 40), (30, 60), (50, 80), (70, 100), (100, 130), (140, 170), (180, 210)]
train_source_path = './bpe_dataset/train_set/source_'
train_target_path = './bpe_dataset/train_set/target_'
valid_source_path = './bpe_dataset/valid_set/source_'
valid_target_path = './bpe_dataset/valid_set/target_'
test_source_path = './bpe_dataset/test_set/source_'
test_target_path = './bpe_dataset/test_set/target_'

bpe2idx_path = './npy/bpe2idx.npy'
idx2bpe_path = './npy/idx2bpe.npy'

saver_path = './saver/'
tensorboard_path = './tensorboard/'

def load_data(path, mode=None):
	data = np.load(path, encoding='bytes')
	if mode == 'dictionary':
		data = data.item()
	return data

def _read_csv(path):
	print('read csv data', path)
	data = np.loadtxt(
			path, 
			delimiter=",", 
			dtype=np.int32,
			ndmin=2 # csv가 1줄이여도 2차원으로 출력.
		)
	return data

def _read_txt(path):
	print('read txt data', path)
	data = []
	with open(path, 'r', encoding='utf-8') as f:
		for sentence in f:
			# EOF check
			if sentence == '\n' or sentence == ' ' or sentence == '':
				break
			if sentence[-1] == '\n':
				sentence = sentence[:-1]
			data.append(sentence.split())					
		return data

def read_data_set(sentence_path, target_path, bucket, target_type='csv'):
	dictionary = {}
	for bucket_size in bucket:
		sentence = _read_csv(sentence_path+str(bucket_size)+'.csv')
		
		if target_type == 'csv':
			target = _read_csv(target_path+str(bucket_size)+'.csv')
		else:
			target = _read_txt(target_path+str(bucket_size)+'.txt')

		# 개수가 0인 bucket은 버림.
		if len(sentence) != 0:
			dictionary[bucket_size] = [sentence, target]
		
			if target_type =='csv':
				print(sentence.shape, target.shape, '\n')
			else:
				print(sentence.shape, len(target), '\n')

	print('\n\n')
	return dictionary
		


def train(model, data, lr):
	loss = 0

	data.shuffle()
	for i in tqdm(range(data.total_iter), ncols=50):
		
		encoder_input, temp, zz = data.get_batch()
		decoder_input = temp[:, :-1] 
		target = temp[:, 1:] # except '</g>'		
		train_loss, _ = sess.run([model.train_cost, model.minimize], 
				{
					model.lr:lr,
					model.encoder_input:encoder_input, 
					model.decoder_input:decoder_input,
					model.target:target, 
					model.keep_prob:0.9 # dropout rate = 0.1		
				}
			)
		loss += train_loss
	
	return loss/data.total_iter


def infer(model, data):
	pred_list = []
	target_list = []
	for i in tqdm(range(data.total_iter), ncols=50):
		encoder_input, target, bucket_size = data.get_batch()
		target_length = bucket_size[1]

		pred = infer_helper.decode(encoder_input, target_length) # [N, target_length]
		first_eos = np.argmax(pred == bpe2idx['</e>'], axis=1) # [N] 최초로 eos 나오는 index.

		for _pred, _first_eos, _target in (zip(pred, first_eos, target)):
			if _first_eos != 0:
				_pred = _pred[:_first_eos]
			_pred = [idx2bpe[idx] for idx in _pred] # idx2bpe
			_pred = ''.join(_pred) # 공백 없이 전부 concat
			_pred = _pred.replace('</w>', ' ') # 공백 symbol을 공백으로 치환.
			pred_list.append(_pred.split())
			target_list.append([_target])
		
	bleu = utils.bleu(target_list, pred_list) * 100
	return bleu



def run(model, trainset, validset, testset, restore=0):
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")

	with tf.name_scope("tensorboard"):
		train_loss_tensorboard = tf.placeholder(tf.float32, name='train_loss')
		valid_bleu_tensorboard = tf.placeholder(tf.float32, name='valid_bleu')
		test_bleu_tensorboard = tf.placeholder(tf.float32, name='test_bleu')

		train_summary = tf.summary.scalar("train_loss", train_loss_tensorboard)
		valid_summary = tf.summary.scalar("valid_bleu", valid_bleu_tensorboard)
		test_summary = tf.summary.scalar("test_bleu", test_bleu_tensorboard)
				
		merged_train_valid = tf.summary.merge([train_summary, valid_summary])
		merged_test = tf.summary.merge([test_summary])
		#merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_path, sess.graph)


	if not os.path.exists(saver_path):
		print("create save directory")
		os.makedirs(saver_path)
	
	save_epoch = 20
	for epoch in range(restore+1, 20000+1):
		lr = (embedding_size**-0.5) * min( (epoch**-0.5), (epoch*(warmup_steps**-1.5)) )

		train_loss = train(model, trainset, lr)
		valid_bleu = infer(model, validset)
	
		summary = sess.run(merged_train_valid, {
					train_loss_tensorboard:train_loss, 
					valid_bleu_tensorboard:valid_bleu,
				}
			)		
		writer.add_summary(summary, epoch)

		if epoch % save_epoch == 0:
			test_bleu = infer(model, testset)
			# tensorboard
			summary = sess.run(merged_test, {
						test_bleu_tensorboard:test_bleu, 
					}
				)
			writer.add_summary(summary, epoch)

			print("epoch:", epoch, 'train_loss:', train_loss,  'valid_bleu:', valid_bleu, 'test_bleu:', test_bleu, 'lr:', lr, '\n')

		else:
			print("epoch:", epoch, 'train_loss:', train_loss,  'valid_bleu:', valid_bleu, 'lr:', lr, '\n')
		


		if (epoch) % save_epoch == 0:
			model.saver.save(sess, saver_path+str(epoch)+".ckpt")

		



print('Data read') # key: bucket_size(tuple) , value: [source, target]
train_dict = read_data_set(train_source_path, train_target_path, bucket)
valid_dict = read_data_set(valid_source_path, valid_target_path, bucket, 'txt')
test_dict = read_data_set(test_source_path, test_target_path, bucket, 'txt')

train_set = bucket_data_helper.bucket_data(train_dict, iter=True, batch_token = 25000) # batch_token // len(sentence||target token) == batch_size
valid_set = bucket_data_helper.bucket_data(valid_dict, iter=True, batch_token = 25000) # batch_token // len(sentence||target token) == batch_size
test_set = bucket_data_helper.bucket_data(test_dict, iter=True, batch_token = 25000) # batch_token // len(sentence||target token) == batch_size

bpe2idx = load_data(bpe2idx_path, mode='dictionary')
idx2bpe = load_data(idx2bpe_path, mode='dictionary')


print("Model read")
sess = tf.Session()

warmup_steps = 4000 #//10
embedding_size = 256#256
encoder_decoder_stack = 3

model = transformer.Transformer(
		sess = sess,
		voca_size = len(bpe2idx), 
		embedding_size = embedding_size, 
		is_embedding_scale = True, 
		PE_sequence_length = 300,
		encoder_decoder_stack = encoder_decoder_stack,
		multihead_num = 8,
		go_idx=bpe2idx['</g>'], 
		eos_idx=bpe2idx['</e>'], 
		pad_idx=bpe2idx['</p>'],
		label_smoothing=0.1
	)
infer_helper = inference_helper.greedy(sess, model, bpe2idx['</g>'])
utils = inference_helper.utils()

print('run')
run(model, train_set, valid_set, test_set)

#print(sess.run(model.embedding_table))

#testcode()


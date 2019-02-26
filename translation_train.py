import tensorflow as tf
import numpy as np
import transformer
import inference_helper
import bucket_data_helper
import os
from tqdm import tqdm
import warnings
import argparse

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

parser = argparse.ArgumentParser()
parser.add_argument(
		'-train_path_2017', 
		help="train_path",
		required=True
	)
parser.add_argument(
		'-valid_path_2014', 
		help="valid_path",
		required=True
	)
parser.add_argument(
		'-test_path_2015', 
		help="test_path",
		required=True
	)
parser.add_argument(
		'-test_path_2016', 
		help="test_path",
		required=True
	)
parser.add_argument(
		'-voca_path', 
		help="Vocabulary_path",
		required=True
	)

args = parser.parse_args()
train_path_2017 = args.train_path_2017
valid_path_2014 = args.valid_path_2014
test_path_2015 = args.test_path_2015
test_path_2016 = args.test_path_2016
voca_path = args.voca_path


saver_path = './saver/'
tensorboard_path = './tensorboard/'


def read_voca(path):
	sorted_voca = []
	with open(path, 'r', encoding='utf-8') as f:	
		for bpe_voca in f:
			bpe_voca = bpe_voca.strip()
			if bpe_voca:
				bpe_voca = bpe_voca.split()
				sorted_voca.append(bpe_voca)
	return sorted_voca


def _read_csv(path):
	data = np.loadtxt(
			path, 
			delimiter=",", 
			dtype=np.int32,
			ndmin=2 # csv가 1줄이여도 2차원으로 출력.
		)
	return data

def _read_txt(path):
	with open(path, 'r', encoding='utf-8') as f:
		documents = f.readlines()
	
	data = []
	for sentence in documents:
		data.append(sentence.strip().split())					
	return data


def _get_bucket_name(path):
	bucket = {}
	for filename in os.listdir(path):
		bucket[filename.split('.')[-2].split('_')[-1]] = 1
	return tuple(bucket.keys())


def make_bpe2idx(voca):
	bpe2idx = {'</p>':0, '</UNK>':1, '</g>':2, '</e>':3}	
	idx2bpe = ['</p>', '</UNK>', '</g>', '</e>']
	idx = 4

	for word, _ in voca:
		bpe2idx[word] = idx
		idx += 1
		idx2bpe.append(word)

	return bpe2idx, idx2bpe


def read_data_set(path, target_type='csv'):
	buckets = _get_bucket_name(path)
	
	dictionary = {}
	total_sentence = 0
	
	for i in tqdm(range(len(buckets)), ncols=50):
		bucket = buckets[i] # '(35, 35)' string
		
		source_path = os.path.join(path, 'source_'+bucket+'.csv')
		sentence = _read_csv(source_path)
		
		if target_type == 'csv':
			target_path = os.path.join(path, 'target_'+bucket+'.csv')
			target = _read_csv(target_path)
		else:
			target_path = os.path.join(path, 'target_'+bucket+'.txt')
			target = _read_txt(target_path)

		# 개수가 0인 bucket은 버림.
		sentence_num = len(sentence)
		if sentence_num != 0:
			total_sentence += sentence_num
			
			sentence_bucket, target_bucket = bucket[1:-1].split(',')
			tuple_bucket = (int(sentence_bucket), int(target_bucket))
			dictionary[tuple_bucket] = [sentence, target]

	print('data_path:', path, 'data_size:', total_sentence, '\n')
	return dictionary



def get_lr(embedding_size, step_num):
	'''
	https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf
	step_num(training_steps):  number of iterations, ie. the number of times the optimizer update was run
		This number also equals the number of mini batches that were processed.
	'''
	lr = (embedding_size**-0.5) * min( (step_num**-0.5), (step_num*(warmup_steps**-1.5)) )
	return lr



def train(model, data, epoch):
	loss = 0

	dataset = data.get_dataset(bucket_shuffle=True, dataset_shuffle=True)
	total_iter = len(dataset)

	for i in tqdm(range(total_iter), ncols=50):
		step_num = ((epoch-1)*total_iter)+(i+1)
		lr = get_lr(embedding_size=embedding_size, step_num=step_num) # epoch: [1, @], i:[0, total_iter)

		encoder_input, temp = dataset[i]
		decoder_input = temp[:, :-1] 
		#print(encoder_input.shape, decoder_input.shape, 4*np.multiply(*encoder_input.shape)*512/1000000000,"GB", 4*np.multiply(*decoder_input.shape)*40297/1000000000,'GB')
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
		#if (i+1) % 5000 == 0:
		#	print(i+1,loss/(i+1), 'lr:', lr)

	print('current step_num:', step_num, 'lr:', lr)
	return loss/total_iter


def infer(model, data):
	pred_list = []
	target_list = []

	dataset = data.get_dataset(bucket_shuffle=False, dataset_shuffle=False)
	total_iter = len(dataset)

	for i in tqdm(range(total_iter), ncols=50):
		encoder_input, target = dataset[i]
		target_length = encoder_input.shape[1] + 30

		pred = infer_helper.decode(encoder_input, target_length) # [N, target_length]
		del encoder_input
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



def run(model, trainset2017, validset2014, testset2015, testset2016, restore=0):
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")
		print('restore:', restore)
	

	with tf.name_scope("tensorboard"):
		train_loss_tensorboard_2017 = tf.placeholder(tf.float32, name='train_loss_2017')
		valid_bleu_tensorboard_2014 = tf.placeholder(tf.float32, name='valid_bleu_2014')
		test_bleu_tensorboard_2015 = tf.placeholder(tf.float32, name='test_bleu_2015')
		test_bleu_tensorboard_2016 = tf.placeholder(tf.float32, name='test_bleu_2016')

		train_summary_2017 = tf.summary.scalar("train_loss_wmt17", train_loss_tensorboard_2017)
		valid_summary_2014 = tf.summary.scalar("valid_bleu_newstest2014", valid_bleu_tensorboard_2014)
		test_summary_2015 = tf.summary.scalar("test_bleu_newstest2015", test_bleu_tensorboard_2015)
		test_summary_2016 = tf.summary.scalar("test_bleu_newstest2016", test_bleu_tensorboard_2016)
				
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
		#merged_train_valid = tf.summary.merge([train_summary, valid_summary])
		#merged_test = tf.summary.merge([test_summary])
		

	if not os.path.exists(saver_path):
		print("create save directory")
		os.makedirs(saver_path)
	
	for epoch in range(restore+1, 20000+1):
		#train 
		train_loss_2017 = train(model, trainset2017, epoch)
		
		#save
		model.saver.save(sess, saver_path+str(epoch)+".ckpt")
		
		#validation 
		valid_bleu_2014 = infer(model, validset2014)
		
		#test
		test_bleu_2015 = infer(model, testset2015)
		test_bleu_2016 = infer(model, testset2016)
		print("epoch:", epoch)
		print('train_loss_wmt17:', train_loss_2017, 'valid_bleu_newstest2014:', valid_bleu_2014)
		print('test_bleu_newstest2015:', test_bleu_2015, 'test_bleu_newstest2016:', test_bleu_2016, '\n')
		
		
		#tensorboard
		summary = sess.run(merged, {
					train_loss_tensorboard_2017:train_loss_2017, 
					valid_bleu_tensorboard_2014:valid_bleu_2014,
					test_bleu_tensorboard_2015:test_bleu_2015, 
					test_bleu_tensorboard_2016:test_bleu_2016, 
				}
			)		
		writer.add_summary(summary, epoch)
		


print('Data read') # key: bucket_size(tuple) , value: [source, target]
train_dict_2017 = read_data_set(train_path_2017)
valid_dict_2014 = read_data_set(valid_path_2014, 'txt')
test_dict_2015 = read_data_set(test_path_2015, 'txt')
test_dict_2016 = read_data_set(test_path_2016, 'txt')

train_set_2017 = bucket_data_helper.bucket_data(train_dict_2017, batch_token = 10000) # batch_token // len(sentence||target token) == batch_size
valid_set_2014 = bucket_data_helper.bucket_data(valid_dict_2014, batch_token = 9000) # batch_token // len(sentence||target token) == batch_size
test_set_2015 = bucket_data_helper.bucket_data(test_dict_2015, batch_token = 9000) # batch_token // len(sentence||target token) == batch_size
test_set_2016 = bucket_data_helper.bucket_data(test_dict_2016, batch_token = 9000) # batch_token // len(sentence||target token) == batch_size
del train_dict_2017, valid_dict_2014, test_dict_2015, test_dict_2016


print("Model read")
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

voca = read_voca(voca_path)
bpe2idx, idx2bpe = make_bpe2idx(voca)
warmup_steps = 4000 * 8 # paper warmup_steps: 4000(with 8-gpus), so warmup_steps of single gpu: 4000*8
embedding_size = 512
encoder_decoder_stack = 6
multihead_num = 8
label_smoothing = 0.1
beam_width = 4
length_penalty = 0.6

print('voca_size:', len(bpe2idx))
print('warmup_steps:', warmup_steps)
print('embedding_size:', embedding_size)
print('encoder_decoder_stack:', encoder_decoder_stack)
print('multihead_num:', multihead_num)
print('label_smoothing:', label_smoothing)
print('beam_width:', beam_width)
print('length_penalty:', length_penalty, '\n')

model = transformer.Transformer(
		sess = sess,
		voca_size = len(bpe2idx), 
		embedding_size = embedding_size, 
		is_embedding_scale = True, 
		PE_sequence_length = 300,
		encoder_decoder_stack = encoder_decoder_stack,
		multihead_num = multihead_num,
		eos_idx=bpe2idx['</e>'], 
		pad_idx=bpe2idx['</p>'],
		label_smoothing=label_smoothing
	)

# beam search
infer_helper = inference_helper.beam(
		sess = sess, 
		model = model, 
		go_idx = bpe2idx['</g>'], 
		eos_idx = bpe2idx['</e>'], 
		beam_width = beam_width, 
		length_penalty = length_penalty
	)
# bleu util
utils = inference_helper.utils()


print('run')
run(model, train_set_2017, valid_set_2014, test_set_2015, test_set_2016)

'''
# greedy search
infer_helper = inference_helper.greedy(
		sess = sess, 
		model = model, 
		go_idx = bpe2idx['</g>']
	)
'''
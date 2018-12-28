import tensorflow as tf
import numpy as np
import transformer
import inference_helper
import os
import bucket_data_helper
import csv
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

test_set_path = './bpe_dataset/testset.csv'
bpe2idx_path = './bpe_dataset/bpe2idx.npy'
idx2bpe_path = './bpe_dataset/idx2bpe.npy'

#test_set_path = './calc/test_set.csv'
saver_path = './saver/'



def load_dictionary(path):
	data = np.load(path, encoding='bytes').item()
	return data


def read_csv(path):
	data = []
	with open(path, 'r', newline='') as f:
		wr = csv.reader(f)
		for sentence in wr:
			data.append(sentence)
	return data


def bucketing(sentence, bucket):
	for bucket_list in bucket:
		if len(sentence) <= bucket_list[0]:	
			sentence = np.pad(
						sentence, 
						(0, bucket_list[0]-len(sentence)),
						'constant',
						constant_values = bpe2idx['</p>'] # pad value
					)
			return sentence, bucket_list[1]#target_length(=decoding step)


def test(model, data, bucket, infer_helper):
	with open('testset_translation.txt', 'w', encoding='utf-8') as o:
		for sentence in tqdm((data), ncols=50):
		#for sentence in tqdm(range( len(data) ), ncols=50):
			sentence = np.array(sentence, dtype=np.int32).reshape(1,-1)
			#print(sentence)
			sentence, target_length = bucketing(sentence, bucket)

			pred = infer_helper.decode(sentence, target_length)[0] # [target_length]
			first_eos = np.argmax(pred == bpe2idx['</e>']) # 최초로 eos 나오는 index.
			if first_eos != 0:
				pred = pred[:first_eos]
			pred_sentence = [idx2bpe[idx] for idx in pred] # idx2bpe
			pred_sentence = ''.join(pred_sentence) # 공백 없이 전부 concat
			pred_sentence = pred_sentence.replace('</w>', ' ') # 공백 symbol을 공백으로 치환.
			#print(pred_sentence)
			o.write(pred_sentence+'\n')


def run(model, testset, bucket, infer_helper, restore=0):
	if restore != 0:
		model.saver.restore(sess, saver_path+str(restore)+".ckpt")

	test(model, testset, bucket, infer_helper)






print('Data read')
test_set = read_csv(test_set_path)
bpe2idx = load_dictionary(bpe2idx_path)
idx2bpe = load_dictionary(idx2bpe_path)

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
infer_helper = inference_helper.greedy(sess, model, bpe2idx['</g>'])

bucket = [(10, 30), (20, 40), (50, 70), (80, 100), (110, 140), (150, 170), (180, 200)]

print('run')
restore = 1100
run(model, test_set, bucket, infer_helper, restore)

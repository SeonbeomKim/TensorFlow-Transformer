import inference_helper 



def read_txt(path, mode='pred'):
	data = []
	with open(path, 'r', newline='', encoding='utf-8') as f:
		for sentence in f:
			split = sentence[:-1].split()
			if mode=='target':
				split = [split]
			data.append(split)
	return data

util = inference_helper.utils()

pred = read_txt('testset_translation.txt')
pred2 = read_txt('testset_translation_beam2.txt')
target = read_txt('test_target.txt', mode='target')

bleu = 0

score = util.bleu(target, pred)
print(score*100)


score = util.bleu(target, pred2)
print(score*100)


'''
count = 0
for i in range(len(pred)):
	score = util.bleu(pred[i], target[i])
	#if score == 1.0:
	#	print(i+1)

	if score > 1e-50:
		count += 1
		bleu += score
		#print(score)
	#bleu += score

bleu /= count
print('count', count)
print('total', len(pred))
#bleu /= len(pred)
bleu *= 100
print(bleu)
'''
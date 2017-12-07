import random
import numpy as np 

"""随机选择非垃圾数据进行扩增

	label - 0 : 不重复上采样
"""


def obtainNegSampleSets(filename, outputFN):
	"""获取正样本比负样本多了几个样本 pos_num - neg_num

		及 负样本集合

	"""
	neg_sample_num, pos_sample_num = 0, 0
	neg_sample = []

	f = open(filename)
	for eachline in f:
		label, _ = eachline.split('\t', 1)

		if label == '0':
			neg_sample_num += 1
			neg_sample.append(eachline)
		elif label == '1':
			pos_sample_num += 1

	f.close()

	diff = pos_sample_num - neg_sample_num

	return diff, neg_sample

def main(filename, outputFN):
	"""主程序"""

	diff, neg_sample = obtainNegSampleSets(filename, outputFN)
	if diff <= 0:
		print("样本数有误！")
		assert 0

	f = open(filename).readlines()
	length = len(neg_sample) # 负样本集合的长度

	while diff > 0:
		randnum = np.random.randint(0, length)
		f.append(neg_sample[randnum])
		neg_sample[randnum], neg_sample[length-1] = neg_sample[length-1], neg_sample[randnum]
		length -= 1
		diff -= 1

	f = random.sample(f, len(f)) # 随机打乱顺序

	with open(outputFN, 'w') as g:
		g.writelines(f)

	print("Finished!")


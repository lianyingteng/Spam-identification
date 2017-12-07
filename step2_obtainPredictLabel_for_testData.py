"""为测试数据预测，得到预测标签

		假定测试数据有标签
"""

def load_model_file(sub_num):
	"""从保存的模型中载入 stop word、 Tfidf_vectorizer、 ont hot编码器
	"""
	with open("model\\stopWordDoc_%d.pkl"%sub_num, 'rb') as stw:
		stop_word_dict = pickle.load(stw)
	with open("model\\Tfidf_vectorizer_%d.pkl"%sub_num, 'rb') as f:
		Tfidf_vectorizer = pickle.load(f)
	with open("model\\OneHotEncoder_%d.pkl"%sub_num, 'rb') as enc:
		encoder = pickle.load(enc)

	return stop_word_dict, Tfidf_vectorizer, encoder


def addNewFeature_1(sample_lists):
	"""特征提取子程序，新增两部分
	
	1. 查找符号【 出现的次数
	2. 款前后都不是汉字计数
	"""
	sample_len = len(sample_lists)
	add_X = np.zeros((sample_len, 2))
	for i in range(sample_len):
		sample = sample_lists[i]

		add_X[i][0] += (sample.count("【") + sample.count("】"))
		add_X[i][1] += len(re.findall(r"\W款\W", sample))

	return add_X

def feature_extract_testSet(sample_lists, sub_num):
	"""参数
        ---
        cv_file: str 
            垃圾短信 验证或测试集 text文件 (无标签数据)

        返回值
        ---
            X: [' ',' ',……]
                所有样本的词string （词与词之间用 空格 分隔）
            y: []
                所有样本的类别标签
    """

	# 载入模型（stop word、 Tfidf_vectorizer、 ont hot编码器）
	stop_word_dict, Tfidf_vectorizer, encoder = load_model_file(sub_num)

	texts = []

	for eachline in sample_lists:
		string = eachline.strip('\n')

		string = ' '.join(
			filter(lambda asd: stop_word_dict.get(asd)==None, jieba.cut(string)
				)
			)

		texts.append(string)


	# String -> feature vector


	X = Tfidf_vectorizer.transform(texts).toarray() #将原始文本统计成TF-IDF值
	# 提取新特征 并标准化
	add_X = addNewFeature_1(sample_lists)
	#print(add_X)
	add_X = encoder.transform(add_X).toarray()
	
	# 合并 两类特征
	X = np.hstack((X, add_X))
	
	return X

def argsParser():
	parser = argparse.ArgumentParser() 
	parser.add_argument("-t",type=str,default="test.txt",help="测试文件名 - 支持相对路径")
	parser.add_argument("-o", type=str, default="output.txt", help="输出文件名")
	args = parser.parse_args()

	return args.t, args.o


import re
import pickle
import jieba
import argparse
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

if __name__ == '__main__':

	test_file, out_preLabel_file = argsParser()


	all_samples = open(test_file).readlines()
	length = len(all_samples)
	pre_label = np.array([2]*length, dtype='<U1')[:, np.newaxis]
	outPreProbability = np.zeros(length)
	invaild_model = 0 # 无效模型

	subClasser_num = 50
	pre_pros = []###
	for i in range(subClasser_num):
		print("第%d个模型"%(i+1))
		with open("model\\LRmodel_%d.pkl"%i, 'rb') as f:
			model = pickle.load(f)

		try:
			X = feature_extract_testSet(all_samples, i)
		except:
			invaild_model += 1
			continue

		p_y = model.predict(X) # 得到预测标签
		pre_label = np.hstack((pre_label, p_y[:,np.newaxis]))
		pre_pro = model.predict_proba(X)[:, 1]
		pre_pros.append(float(pre_pro)) ###
		outPreProbability += pre_pro

	print(list(pre_label))
	pre_label = list(map(lambda asd: '1' if list(asd).count('1') > list(asd).count('0') else '0', pre_label))
	outPreProbability = outPreProbability/(50-invaild_model)
	# print(pre_label)
	pre_pros.sort()
	print(pre_pros)####
	# 生成 有标签数据 文档
	with open(out_preLabel_file, 'w') as f:
		for i in range(len(all_samples)):
			f.write("%s\t%f\t%s"%(p_y[i], outPreProbability[i], all_samples[i]))

	print("Finished!")
	print("无效模型数： %d\\%d"%(invaild_model, subClasser_num))

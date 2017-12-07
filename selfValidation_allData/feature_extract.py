"""
		构建特征向量
"""
import pickle
import jieba
import random
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression

def obtain_stop_word_dict(stopWordFile):
	"""从停止词pkl文件中载入全部的 stop word

		return： 
		---
		停止词字典dict 
			key -> word
	"""
	stop_word_file = stopWordFile
	f = open(stop_word_file, 'rb')
	stop_word_dict = pickle.load(f)
	f.close()

	return stop_word_dict


def feature_extract(sample_lists, stopWordFile):
	"""参数
        ---
        train_file: str
            垃圾短信text文件

        返回值
        ---
            X: [' ',' ',……]
                所有样本的词string （词与词之间用 空格 分隔）
            y: []
                所有样本的类别标签
	"""
	print("feature extraction...")

	# 获得停止词
	stop_word_dict = obtain_stop_word_dict(stopWordFile)

	y, texts = [], []

	for eachline in sample_lists:
		try:
			label, string = eachline.strip('\n').split('\t', 1)
		except:
			continue

		y.append(label)

		string = ' '.join(
			filter(lambda asd: stop_word_dict.get(asd)==None, jieba.cut(string)
				)
			)

		texts.append(string)


	# String -> feature vector
	print("\ttf-idf encoding......")
	Tfidf_vectorizer = TfidfVectorizer() #建立 tf-idf 特征生成器
	Tfidf_vectorizer.fit(texts) #  拟合 (建模时应该保存)

	X = Tfidf_vectorizer.transform(texts) #将原始文本统计成TF-IDF值

	print("feature extraction has finished!")

	return X, y

def main(train_file, outputFile, stopWordFile):
	"""自查自检主程序

	生成预测错误的样本文件 error_classify_sample.txt


	"""
	all_samples = open(train_file).readlines() # 所有样本
	sample_len = len(all_samples) # 样本长度
	all_samples = random.sample(all_samples, sample_len) #  随机重新排序

	# 11. 自检自查 用于查找 原始数据中的噪音
	X, y = feature_extract(all_samples, stopWordFile)
	model = LogisticRegression(penalty='l2', C=80)
	model.fit(X, y)
	p_y = model.predict(X)

	f = open(outputFile, 'w')
	ind = 0
	for pre, ori in zip(p_y, y):
		if pre != ori:
			f.write(all_samples[ind])
		ind += 1

	f.close()

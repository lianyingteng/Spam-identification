def checkDirectoryExists():
	"""检查temp目录是否存在
-
	存在的话 - 删除并创建， 不存在的话 - 创建
	"""
	tempdir = "model"
	if os.path.isdir(tempdir):
		shutil.rmtree(tempdir)

	os.mkdir(tempdir)


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

def feature_extract(sample_lists, stopWordFile, sub_num):
	"""特征提取主程序

	参数
        ---
        sample_lists: [str , str, ..., str]
            训练集 样本的list
        
        stopWordFile： str
        	停用词文件名

        sub_num： int
        	子分类器的索引（第 i 个）

    返回值
    ---
        X: [' ',' ',……]
           所有样本的词string （词与词之间用 空格 分隔）
        y: []
           所有样本的类别标签
	"""
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
	Tfidf_vectorizer = TfidfVectorizer() #建立 tf-idf 特征生成器
	Tfidf_vectorizer.fit(texts) #  拟合 (建模时应该保存)
	X = Tfidf_vectorizer.transform(texts).toarray() #将原始文本统计成TF-IDF值
	# 提取新特征 并标准化
	add_X = addNewFeature_1(sample_lists)
	encoder = OneHotEncoder()
	encoder.fit(add_X)
	add_X = encoder.transform(add_X).toarray()
	# 合并 两类特征
	print(X.shape)
	print(add_X.shape)
	X = np.hstack((X,add_X))
	print(X.shape)

	# 保存 标准化生成器 和  TF-idf 生成器
	with open("model\\OneHotEncoder_%d.pkl"%sub_num, 'wb') as enc:
		pickle.dump(encoder, enc)
	with open("model\\Tfidf_vectorizer_%d.pkl"%sub_num, 'wb') as k:
		pickle.dump(Tfidf_vectorizer, k)

	print("feature extraction has finished!")

	return X, y

def generateStopWordsText(new_sample_set, sub_num):
	"""生成stop word文件，并返回其文件名
	"""
	stopWord_file = "model\\stopWordDoc_%d.pkl"%sub_num

	y, texts = [], []
	for eachline in new_sample_set:
		label, string = eachline.strip('\n').split('\t', 1)

		y.append(label)
		texts.append(' '.join(jieba.cut(string)))
	# String -> feature vector
	Tfidf_vectorizer = TfidfVectorizer() #建立 tf-idf 特征生成器
	Tfidf_vectorizer.fit(texts) #  拟合 (建模时应该保存)
	X = Tfidf_vectorizer.transform(texts)

	words_dict = Tfidf_vectorizer.vocabulary_  # 词位置dict
	words_list = list(map(lambda wc: wc[0], sorted(words_dict.items(), key=lambda asd:asd[1], reverse=False)))
	f_score, p_val = f_classif(X, y) # Anova

	stopword = dict()
	for i in range(len(words_list)):
		
		if (f_score[i] <= 1.0 or p_val[i] > 0.05):
			stopword[words_list[i]] = 1 

	f = open(stopWord_file, 'wb')
	pickle.dump(stopword, f)
	f.close()

	return stopWord_file


def build_Model_basedon_bagging(train_file):
	"""	创建最终的模型 model（们）， 基于随机森林的思想

	参数： str
		训练集文件 名
	"""
	all_samples = open(train_file).readlines() # 所有样本
	length = len(all_samples)

	subClasser_num = 50 # 基本分类器的数量

	for i in range(subClasser_num): # 第i个子分类器
		print("正在生成第%d个模型"%(i+1))
		new_sample_set = []
		for _ in range(length):
			new_sample_set.append(all_samples[np.random.randint(length)])

		stopWord_file = generateStopWordsText(new_sample_set, i)
		X, y = feature_extract(new_sample_set, stopWord_file, i) # 训练集提取特征

		model = LogisticRegression(penalty='l2', C=20)
		model.fit(X, y)

		with open("model\\LRmodel_%d.pkl"%i, 'wb') as f:
			pickle.dump(model, f)

	print("Finished!")

def argsParser():
	parser = argparse.ArgumentParser() 
	parser.add_argument(
        "-t",
        type=str,
        default="train.txt",
        help="训练文件名 - 支持相对路径",
    )
	args = parser.parse_args()

	return args.t


import re
import os
import shutil
import argparse
import pickle
import jieba
import random
import numpy as np
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
	
	train_file = argsParser()

	checkDirectoryExists()
	build_Model_basedon_bagging(train_file)
	parser = argparse.ArgumentParser() 
	parser.add_argument("-t", help="训练文件名 - 支持相对路径")
	args = parser.parse_args()
"""
	构建特征向量
"""

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



def feature_extract_testSet(sample_lists, stopWordFile):
	"""参数
        ---
        cv_file: str
            垃圾短信 验证或测试集 text文件

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

	# 载入 标准化生成器 和  TF-idf 生成器
	with open("Tfidf_vectorizer.pkl", 'rb') as f:
		Tfidf_vectorizer = pickle.load(f)
	with open("OneHotEncoder.pkl", 'rb') as enc:
		encoder = pickle.load(enc)

	X = Tfidf_vectorizer.transform(texts).toarray() #将原始文本统计成TF-IDF值
	# 提取新特征 并标准化
	print("\tone hot encoding......")
	add_X = addNewFeature_1(sample_lists)
	print(add_X)
	add_X = encoder.transform(add_X).toarray()
	
	# 合并 两类特征
	X = np.hstack((X, add_X))
	print("feature extraction has finished!")

	return X, y


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
	X = Tfidf_vectorizer.transform(texts).toarray() #将原始文本统计成TF-IDF值
	# 提取新特征 并标准化
	print("\tont hot encoding......")
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
	with open("OneHotEncoder.pkl", 'wb') as enc:
		pickle.dump(encoder, enc)
	with open("Tfidf_vectorizer.pkl", 'wb') as k:
		pickle.dump(Tfidf_vectorizer, k)

	print("feature extraction has finished!")

	return X, y


def step1_learning_curve(train_file, stopWordFile):
	"""步骤1. 绘制学习曲线
		
		损失函数随样本量m的变化曲线
	"""

	all_samples = open(train_file).readlines() # 所有样本
	sample_len = len(all_samples) # 样本长度

	train_set_sizes = list(
		map(lambda asd: int(asd), np.linspace(1, sample_len, 300))
		)[1:]  # 定义训练集的大小 递增

	train_scores = []
	test_scores = []
	count = 0 # 计数
	for train_size in train_set_sizes:
		count += 1
		print('\n');print(count)
		
		train_sample_set = random.sample(all_samples, train_size) # 随机选择train_size个训练样本

		X, y = feature_extract(train_sample_set, stopWordFile)
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)

		model = LogisticRegression(penalty='l2')
		train_score, test_score = validation_curve(
			model, X, y, param_name="C", param_range=[100000000],
			cv=5, scoring="neg_log_loss"
		)
		train_scores.append(np.mean(train_score))
		test_scores.append(np.mean(test_score))

	# 画图
	plt.title("Validation Curve with LogisticRegression")
	plt.xlabel("Train Size m")
	plt.ylabel("neg_log_loss")

	plt.plot(train_set_sizes, train_scores, 'go-',label='train_scores', linewidth=2)
	plt.plot(train_set_sizes, test_scores, 'ro-',label='test_scores', linewidth=2)
	plt.legend(loc="best")
	plt.show()


def step2_crossValidation_curve(train_file, stopWordFile):
	"""步骤2. 绘制验证曲线

	ACC 随 正则化参数的 变化曲线
	"""

	all_samples = open(train_file).readlines() # 所有样本
	X, y = feature_extract(all_samples, stopWordFile)
	#param_range = list(map(lambda asd: int(asd)/100, np.logspace(0, 10, 11, base=2)))
	param_range = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20, 40, 80, 160]

	model = LogisticRegression(penalty='l2')

	train_scores, test_scores = validation_curve(
		model, X, y, param_name="C", param_range=param_range,
		cv=5, scoring="accuracy"
		)

	train_scores_mean = np.mean(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)

	# 画图
	temp = list(test_scores_mean)
	max_i = temp.index(max(temp))
	print(param_range[max_i])
	print(test_scores_mean[max_i])

	plt.title("Validation Curve with LogisticRegression")
	plt.xlabel("C value")
	plt.ylabel("Accuracy")
	plt.xlim(-2, 180)

	plt.plot(param_range, train_scores_mean, 'go-',label='train_scores', linewidth=2)
	plt.plot(param_range, test_scores_mean, 'ro-',label='test_scores', linewidth=2)

	plt.scatter(param_range[max_i], test_scores_mean[max_i], s=100, color='b')

	plt.legend(loc="best")
	plt.show()
	print(test_scores_mean)


def step3_errorClassSimple_for_testSet(train_file, stopWordFile, test_size, outputFile):
	"""步骤3： 查看验证集错误分类的样本，（用于发现新特征）

	将训练数据集分为 训练集 和 验证集， 用训练集训练模型， 用验证集测试，输出验证集错误分类的样本（文件）

	"""
	all_samples = open(train_file).readlines() # 所有样本
	sample_len = len(all_samples) # 样本长度
	all_samples = random.sample(all_samples, sample_len) #  随机重新排序

	test_end_loc = int(sample_len * test_size) # 测试集最终索引位置

	validate_sample = all_samples[: test_end_loc]
	
	
	X, y = feature_extract(all_samples[test_end_loc :], stopWordFile) # 训练集提取特征
	model = LogisticRegression(penalty='l2', C=20)
	model.fit(X, y)
	
	X, y = feature_extract_testSet(validate_sample, stopWordFile) # 验证集 提取特征
	p_y = model.predict(X)
	p_y_score = model.predict_proba(X)

	print("该模型的准确率（Accuary）: %f"%(accuracy_score(y, p_y)))
	print("该模型的AUC值 : %f"%(roc_auc_score(y, p_y_score)))

	f = open(outputFile, 'w')
	ind = 0
	for pre, ori in zip(p_y, y):
		if pre != ori:
			f.write(validate_sample[ind])
		ind += 1

	f.close()

	print("finished！")


def step4_findErrorClassSample(train_file, stopWord_file, output_file, test_size):
	"""查找错误分类的样本 ， 并将其保存到文件中

		用于校正样本

	"""
	all_samples = open(train_file).readlines() # 所有样本
	sample_len = len(all_samples) # 样本长度
	all_samples = random.sample(all_samples, sample_len) #  随机重新排序

	test_end_loc = int(sample_len * test_size) # 测试集最终索引位置

	validate_sample = all_samples[: test_end_loc]
	
	
	X, y = feature_extract(all_samples[test_end_loc :], stopWord_file) # 训练集提取特征
	model = LogisticRegression(penalty='l2', C=20)
	model.fit(X, y)
	
	X, y = feature_extract_testSet(validate_sample, stopWord_file) # 验证集 提取特征
	p_y = model.predict(X)
	p_y_score = model.predict_proba(X)

	print("该模型的准确率（Accuary）: %f"%(accuracy_score(y, p_y)))
	
	print("该模型的AUC值 : %f"%(roc_auc_score(np.array(list(map(lambda asd: int(asd), y))), (p_y_score.T)[1])))

	f = open(output_file, 'w')
	ind = 0
	for pre, ori in zip(p_y, y):
		if pre != ori:
			f.write(validate_sample[ind])
		ind += 1

	f.close()

	print("finished！")


def step5_build_Model(train_file, stopWord_file):
	"""	创建最终的模型 model

		保存到文件中

	"""
	all_samples = open(train_file).readlines() # 所有样本
	
	X, y = feature_extract(all_samples, stopWord_file) # 训练集提取特征
	model = LogisticRegression(penalty='l2', C=20)
	model.fit(X, y)

	with open("LRmodel.pkl", 'wb') as f:
		pickle.dump(model, f)

	print("Finished!")
	


import re
import pickle
import jieba
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
	
	train_file = "train.txt"
	stopWord_file = "__temp__\\stopWords.pkl"
	output_file = "error_classify_sample.txt" # 验证集中误分类样本集合
	test_size = 0.3 # 前 30% 为 测试集（验证集)

	
	#step1_learning_curve(train_file, stopWord_file) # 学习曲线
	step2_crossValidation_curve(train_file, stopWord_file) # 验证曲线
	#step3_errorClassSimple_for_testSet(train_file, stopWord_file, test_size, output_file) # 输出误分类样本数据
	#step4_findErrorClassSample(train_file, stopWord_file, output_file, test_size)
	#step5_build_Model(train_file, stopWord_file)




import jieba
from sklearn.feature_selection import f_classif
from sklearn.feature_extraction.text import TfidfVectorizer

def obtainStopWordsMain(X, y, words_list, output_file):
	"""由ANOVA对特征进行打分，形式如下：

		word  F-score  p-val

		选择 F-score <= 1 对应的词为stop word

	"""
	f_score, p_val = f_classif(X, y)

	g = open(output_file,'w')
	for i in range(len(f_score)):
		g.write('%s\t%.6f\t%.6f\n'%(words_list[i], f_score[i], p_val[i]))
	g.close()

	print("文件已经生成！")


	

def main(train_file, output_file):
	"""将样本文本 编辑成 tf-idf 特征

		然后 交由 ANOVA 挑选无用特征词

	"""
	
	y, texts = [], []
	f = open(train_file)
	for eachline in f:
		try:
			label, string = eachline.strip('\n').split('\t', 1)
		except:
			continue

		y.append(label)
		texts.append(' '.join(jieba.cut(string)))
	f.close()


	# String -> feature vector
	print("\ttf-idf encoding......")
	Tfidf_vectorizer = TfidfVectorizer() #建立 tf-idf 特征生成器
	Tfidf_vectorizer.fit(texts) #  拟合 (建模时应该保存)
	X = Tfidf_vectorizer.transform(texts)

	words_dict = Tfidf_vectorizer.vocabulary_  # 词位置dict
	words_list = list(map(lambda wc: wc[0], sorted(words_dict.items(), key=lambda asd:asd[1], reverse=False)))
	obtainStopWordsMain(X, y, words_list, output_file)



if __name__ == '__main__':
	train_file = r"..\train.txt"
	output_file = r"stopWord.txt"
	main()

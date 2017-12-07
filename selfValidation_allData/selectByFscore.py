import pickle

""""
	将特征 F-score <= 1.0 并且 p值 > 0.05 的特征 视为 无效特征
		其对应的词称为 stop word
"""

def main(anovaSortFile, stopWordFile_pre):

	f = open(anovaSortFile)
	g = open(stopWordFile_pre+'.txt','w')

	stopword = dict()

	for eachline in f:
		temp = eachline.split()
		word = temp[0]
		fscore = float(temp[1])
		pval = float(temp[2])
		if (fscore <= 1.0 or pval > 0.05):
			g.write(eachline)
			stopword[word] = 1 

	f.close()
	g.close()

	stopwordFile = stopWordFile_pre + '.pkl'
	f = open(stopwordFile, 'wb')
	pickle.dump(stopword, f)
	f.close()

	return stopwordFile
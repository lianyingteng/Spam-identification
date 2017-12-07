import os
import shutil
import selfValidation_allData.feature_extract as svad_main
import selfValidation_allData.LR_getonehot_wordANDf as obtainFscore
import selfValidation_allData.selectByFscore as obtainStopWords
import selfValidation_allData.addPosSample as upSample


def checkDirectoryExists():
	"""检查temp目录是否存在
-
	存在的话 - 删除并创建， 不存在的话 - 创建
	"""
	tempdir = "__temp__"
	if os.path.isdir(tempdir):
		shutil.rmtree(tempdir)

	os.mkdir(tempdir)




sampleBackupFile = "train_backup.txt"

if __name__ == '__main__':

	checkDirectoryExists()
	
	TrainsampleFile = "train.txt"
	anovaSortFile = r"__temp__\tfidf_fscore_pval.txt"
	stopWordFile_pre = r"__temp__\stopWords"
	errClassTrainsampleFile = r"error_classify_sample.txt"

	upSample.main(sampleBackupFile, TrainsampleFile)
	obtainFscore.main(TrainsampleFile, anovaSortFile)
	stopWordFile = obtainStopWords.main(anovaSortFile, stopWordFile_pre)
	svad_main.main(TrainsampleFile, errClassTrainsampleFile, stopWordFile)
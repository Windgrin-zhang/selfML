# -*-coding:utf-8 -*-
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

"""
Author:
	Alex Zhang
Modify:
	2025-8-1
"""

def loadDataSet(fileName):
	numFeat = len((open(fileName).readline().split('\t')))
	dataMat = []; labelMat = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr = []
		curLine = line.strip().split('\t')
		for i in range(numFeat - 1):
			lineArr.append(float(curLine[i]))
		dataMat.append(lineArr)
		labelMat.append(float(curLine[-1]))

	return dataMat, labelMat

if __name__ == '__main__':
	dataArr, classLabels = loadDataSet('horseColicTraining2.txt')
	testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2), algorithm = "SAMME", n_estimators = 10)
	bdt.fit(dataArr, classLabels)
	predictions = bdt.predict(dataArr)
	errArr = np.asmatrix(np.ones((len(dataArr), 1)))
	print('训练集的错误率:%.3f%%' % float(errArr[predictions != classLabels].sum() / len(dataArr) * 100))
	predictions = bdt.predict(testArr)
	errArr = np.asmatrix(np.ones((len(testArr), 1)))
	print('测试集的错误率:%.3f%%' % float(errArr[predictions != testLabelArr].sum() / len(testArr) * 100))
# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
	"""
	函数说明:加载数据
	Parameters:
		fileName - 文件名
	Returns:
		xArr - x数据集
		yArr - y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

def ridgeRegres(xMat, yMat, lam = 0.2):
	"""
	函数说明:岭回归
	Parameters:
		xMat - x数据集
		yMat - y数据集
		lam - 缩减系数
	Returns:
		ws - Regression Coefficients
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def ridgeTest(xArr, yArr):
	"""
	函数说明:岭回归测试
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		wMat - Regression Coefficients矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	xMat = np.asmatrix(xArr); yMat = np.asmatrix(yArr).T
	#数据标准化
	yMean = np.mean(yMat, axis = 0)						#行与行操作，求均值
	yMat = yMat - yMean									#数据减去均值
	xMeans = np.mean(xMat, axis = 0)					#行与行操作，求均值
	xVar = np.var(xMat, axis = 0)						#行与行操作，求方差
	xMat = (xMat - xMeans) / xVar						#数据减去均值除以方差实现标准化
	numTestPts = 30										#30个不同的lambda测试
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))	#初始Regression Coefficients矩阵
	for i in range(numTestPts):							#改变lambda计算Regression Coefficients
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))	#lambda以e的指数变化，最初是一个非常小的数，
		wMat[i, :] = ws.T 								#计算Regression Coefficients矩阵
	return wMat

def plotwMat():
	"""
	函数说明:绘制岭回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	abX, abY = loadDataSet('abalone.txt')
	redgeWeights = ridgeTest(abX, abY)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(redgeWeights)	
	ax.set_title('log(lambda) vs Regression Coefficients', fontsize=20, weight='bold', color='red')
	ax.set_xlabel('log(lambda)', fontsize=10, weight='bold', color='black')
	ax.set_ylabel('Regression Coefficients', fontsize=10, weight='bold', color='black')
	plt.show()


def regularize(xMat, yMat):
	"""
	函数说明:数据标准化
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		inxMat - 标准化后的x数据集
		inyMat - 标准化后的y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""	
	inxMat = xMat.copy()														#数据拷贝
	inyMat = yMat.copy()
	yMean = np.mean(yMat, 0)													#行与行操作，求均值
	inyMat = yMat - yMean														#数据减去均值
	inMeans = np.mean(inxMat, 0)   												#行与行操作，求均值
	inVar = np.var(inxMat, 0)     												#行与行操作，求方差
	inxMat = (inxMat - inMeans) / inVar											#数据减去均值除以方差实现标准化
	return inxMat, inyMat

def rssError(yArr,yHatArr):
	"""
	函数说明:计算平方误差
	Parameters:
		yArr - 预测值
		yHatArr - 真实值
	Returns:
		
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	return ((yArr-yHatArr)**2).sum()

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
	"""
	函数说明:前向逐步线性回归
	Parameters:
		xArr - x输入数据
		yArr - y预测数据
		eps - 每次迭代需要调整的步长
		numIt - Iterations
	Returns:
		returnMat - numIt次迭代的Regression Coefficients矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	xMat = np.asmatrix(xArr); yMat = np.asmatrix(yArr).T 										#数据集
	xMat, yMat = regularize(xMat, yMat)												#数据标准化
	m, n = np.shape(xMat)
	returnMat = np.zeros((numIt, n))												#初始化numIt次迭代的Regression Coefficients矩阵
	ws = np.zeros((n, 1))															#初始化Regression Coefficients矩阵
	wsTest = ws.copy()
	wsMax = ws.copy()
	for i in range(numIt):															#迭代numIt次
		# print(ws.T)																	#打印当前Regression Coefficients矩阵
		lowestError = float('inf'); 												#正无穷
		for j in range(n):															#遍历每个特征的Regression Coefficients
			for sign in [-1, 1]:
				wsTest = ws.copy()
				wsTest[j] += eps * sign												#微调Regression Coefficients
				yTest = xMat * wsTest												#计算预测值
				rssE = rssError(yMat.A, yTest.A)									#计算平方误差
				if rssE < lowestError:												#如果误差更小，则更新当前的最佳Regression Coefficients
					lowestError = rssE
					wsMax = wsTest
		ws = wsMax.copy()
		returnMat[i,:] = ws.T 														#记录numIt次迭代的Regression Coefficients矩阵
	return returnMat

def plotstageWiseMat():
	"""
	函数说明:绘制前向逐步回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2025-8-1
	"""
	xArr, yArr = loadDataSet('abalone.txt')
	returnMat = stageWise(xArr, yArr, 0.005, 1000)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(returnMat)	
	ax.set_title('Forward Stepwise Regression: Iterations vs Coefficients', fontsize=15, weight='bold', color='red')
	ax.set_xlabel('Iterations', fontsize=10, weight='bold', color='black')
	ax.set_ylabel('Regression Coefficients', fontsize=10, weight='bold', color='black')
	plt.show()


if __name__ == '__main__':
	plotstageWiseMat()
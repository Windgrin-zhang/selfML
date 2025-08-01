import numpy as np
import operator

"""
函数说明:创建数据集

Returns:
	group - 数据集
	labels - 分类标签
Modify:
	2025-7-30
"""
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2025-7-30
"""
def classify0(inX, dataSet, labels, k):
	#numpy函数shape[0]返回dataSet的行数                                                             数个数
	dataSetSize = dataSet.shape[0]
	#在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)                           算各维度差值
	diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
	#二维特征相减后平方                                                                             计算各维度差值的平方
	sqDiffMat = diffMat**2
	#sum()所有元素相加，sum(0)列相加，sum(1)行相加                                                    计算各维度差值的平方和    
	sqDistances = sqDiffMat.sum(axis=1)
	#开方，计算出距离                                                                               计算各维度差值的平方和开方（欧式距离）
	distances = sqDistances**0.5
	#返回distances中元素从小到大排序后的索引值                                                         给出各个点与测试点的距离从小到大排序后的索引值
	sortedDistIndices = distances.argsort()
    #打印排序后的索引值，观察现象
	print(sortedDistIndices)
	#定一个记录类别次数的字典                                                                         记录每个类别最近的次数（类似于投票）
	classCount = {}
	for i in range(k):
		#取出前k个元素的类别                                                                         取出距离最近的k个点的类别（自己设置，基本上小于总数，若总数量较多则取较少的数量）
		voteIlabel = labels[sortedDistIndices[i]]
		#dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
		#计算类别次数
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#python3中用items()替换python2中的iteritems()
	#key=operator.itemgetter(1)根据字典的值进行排序
	#key=operator.itemgetter(0)根据字典的键进行排序
	#reverse降序排序字典                                                                             我们需要的是各value的数量的降序排序
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#返回次数最多的类别,即所要分类的类别
	return sortedClassCount[0][0]

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [101,20]
	#kNN分类
	test_class = classify0(test, group, labels, 4)
	#打印分类结果
	print(test_class)

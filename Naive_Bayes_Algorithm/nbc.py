# -*- coding: UTF-8 -*-
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import os
import random
import jieba                                                    #jieba中文分词

def TextProcessing(folder_path, test_size = 0.2):
	folder_list = os.listdir(folder_path)						#查看folder_path下的文件
	data_list = []												#数据集数据
	class_list = []												#数据集类别

	#遍历每个子文件夹
	for folder in folder_list:
		new_folder_path = os.path.join(folder_path, folder)		#根据子文件夹，生成新的路径
		files = os.listdir(new_folder_path)						#存放子文件夹下的txt文件的列表
		
		j = 1
		#遍历每个txt文件
		for file in files:
			if j > 100:											#每类txt样本数最多100个
				break
			with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:	#打开txt文件
				raw = f.read()
			
			word_cut = jieba.cut(raw, cut_all = False)			#精简模式，返回一个可迭代的generator
			word_list = list(word_cut)							#generator转换为list
			
			data_list.append(word_list)							#添加数据集数据
			class_list.append(folder)							#添加数据集类别
			j += 1

	data_class_list = list(zip(data_list, class_list))			#zip压缩合并，将数据与标签对应压缩
	random.shuffle(data_class_list)								#将data_class_list乱序
	index = int(len(data_class_list) * test_size) + 1			#训练集和测试集切分的索引值
	train_list = data_class_list[index:]						#训练集
	test_list = data_class_list[:index]							#测试集
	train_data_list, train_class_list = zip(*train_list)		#训练集解压缩
	test_data_list, test_class_list = zip(*test_list)			#测试集解压缩

	all_words_dict = {}											#统计训练集词频
	for word_list in train_data_list:
		for word in word_list:
			if word in all_words_dict.keys():
				all_words_dict[word] += 1
			else:
				all_words_dict[word] = 1
	
	#根据键的值倒序排序
	all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
	all_words_list, all_words_nums = zip(*all_words_tuple_list)	#解压缩
	all_words_list = list(all_words_list)						#转换成列表
	return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list


def MakeWordsSet(words_file):
	words_set = set()											#创建set集合
	with open(words_file, 'r', encoding = 'utf-8') as f:		#打开文件
		for line in f.readlines():								#一行一行读取
			word = line.strip()									#去回车
			if len(word) > 0:									#有文本，则添加到words_set中
				words_set.add(word)								
	return words_set 											#返回处理结果


def TextFeatures(train_data_list, test_data_list, feature_words):
	def text_features(text, feature_words):						#出现在特征集中，则置1												
		text_words = set(text)
		features = [1 if word in text_words else 0 for word in feature_words]
		return features
	train_feature_list = [text_features(text, feature_words) for text in train_data_list]
	test_feature_list = [text_features(text, feature_words) for text in test_data_list]
	return train_feature_list, test_feature_list				#返回结果


def words_dict(all_words_list, deleteN, stopwords_set = set()):
	feature_words = []							#特征列表
	n = 1
	for t in range(deleteN, len(all_words_list), 1):
		if n > 1000:							#feature_words的维度为1000
			break								
		#如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
		if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
			feature_words.append(all_words_list[t])
		n += 1
	return feature_words


def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
	classifier = MultinomialNB().fit(train_feature_list, train_class_list)
	test_accuracy = classifier.score(test_feature_list, test_class_list)
	return test_accuracy

if __name__ == '__main__':
	# 创建简单的测试数据
	print("创建测试数据...")
	# 模拟文本分类数据
	test_texts = [
		"这是一个关于科技的文章",
		"科技发展很快",
		"人工智能技术",
		"机器学习算法",
		"深度学习模型",
		"体育新闻",
		"足球比赛",
		"篮球运动",
		"奥运会",
		"体育竞技"
	]
	test_labels = ['科技', '科技', '科技', '科技', '科技', '体育', '体育', '体育', '体育', '体育']
	
	# 简单的文本特征提取
	feature_words = ['科技', '体育', '文章', '发展', '技术', '算法', '模型', '新闻', '比赛', '运动']
	
	# 创建特征矩阵
	train_feature_list = []
	test_feature_list = []
	
	for text in test_texts:
		features = [1 if word in text else 0 for word in feature_words]
		train_feature_list.append(features)
		test_feature_list.append(features)
	
	# 使用sklearn的朴素贝叶斯
	from sklearn.naive_bayes import MultinomialNB
	classifier = MultinomialNB().fit(train_feature_list, test_labels)
	test_accuracy = classifier.score(test_feature_list, test_labels)
	print(f"测试准确率: {test_accuracy:.2f}")
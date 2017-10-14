import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import nltk

class SkipGram():
	def __init__(self,voc_size,hidden_size):
		self.l1 = nn.Linear(voc_size,hidden_size)
		self.sigmoid = nn.Sigmoid()
		self.l2 = nn.Linear(hidden_size,voc_size)
		self.softmax = nn.Softmax()
	def forward(self,x):
		hidden_vector = self.sigmoid(self.l1(x))
		output_vector = self.softmax(self.l2(hidden_vector))

def ReadFile(path):
	data = []
	with open(path) as f:
		for line in f:
			line = line.strip().lower()
			data.append(nltk.word_tokenize(line))
	return data

def ReadDataFromDir(begin,end,vocabulary_size,window_size):
	data = []
	for i in range(begin,end+1):
		data += ReadFile("./data/nyt/file"+str(i)+".txt")

	dic = {}
	for sentence in data:
		for token in sentence:
			if token not in dic:
				dic[token] = 0
			dic[token] += 1

	##Eliminate some tokens
	token_list = []
	for token,count in dic.items():
		token_list.append([token,count])
	token_list = sorted(token_list, key=lambda x:x[1], reverse=True)[0:vocabulary_size-1]
	dic.clear()
	for i in range(len(token_list)):
		if token_list[i][0]=="''":
			token_list[i][0] = "\""
		dic[token_list[i][0]] = i
	dic["UNKNOWNWORD"] = vocabulary_size-1

	##Write Vocabulary
	file = open("./data/vocabulary.txt","w+")
	for token,idx in dic.items():
		file.write(str(token)+" "+str(idx)+"\n")
	file.close()

	##Create the trainable data tuple
	train_file = open("./data/train_file.txt","w+")
	test_file = open("./data/test_file.txt","w+")
	for sentence in data:
		for i in range(window_size,len(sentence)-window_size):
			tmp_rnd = random.randint(1,10)
			if tmp_rnd<=8:
				file = train_file
			else:
				file = test_file

			input_token = sentence[i]
			if input_token not in dic:
				input_token = "UNKNOWNWORD"
			file.write(input_token+" ")

			for context_id in range(i-window_size,i+window_size+1):
				if context_id==i:
					continue
				output_token = sentence[context_id]
				if output_token not in dic:
					output_token = "UNKNOWNWORD"
				file.write(output_token+" ")
			file.write("\n")
	file.close()
	train_file.close()
	test_file.close()

	# data_word = np.array(data_word)
	# data_context = np.array(data_context)

	# np.savetxt("./data/data_word.txt",data_word,delimiter=",")
	# np.savetxt("./data/data_context.txt",data_context,delimiter=",")

	return

def GenerateSample(line,dic):
	input_vector = np.zeros((len(dic),))
	output_vector = np.zeros((len(dic),))

	token = line.strip().split(" ")
	input_vector[dic[token[0]]] = 1
	for i in range(1,len(token)):
		output_vector[dic[token[i]]] = 1
	return input_vector,output_vector

def Train(batch_size):
	train_file = open("./data/train_file.txt")


if __name__=="__main__":
	vocabulary_size = 5000
	window_size = 2
	batch_size = 100
	ReadDataFromDir(0,999,vocabulary_size,window_size)
	# print(len(dic))
	Train(batch_size,vocabulary_size,window_size,hidden_size)
	




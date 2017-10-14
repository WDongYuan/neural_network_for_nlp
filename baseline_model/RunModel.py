from ReadData import *
from Model import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys

def GetOrder(val,arr):
	order = 1
	for i in range(len(arr)):
		if arr[i]>val:
			order += 1
	return order
def save_model(state, filename='saved_model.out'):
    torch.save(state, filename)

def TrainModel(train_data,word_em,D,load_model=""):

	global UNKNOWNWORD
	hidden_size = 200
	embedding_size = D
	epoch_num = 1
	direction = 2
	batch_size = 100

	model = Model(embedding_size,hidden_size,direction,word_em)
	optimizer = optim.SGD(model.parameters(), lr=0.1)

	model_saved_name = ""
	try:
		model_saved_name = model.name+"_"+str(model.embedding_size)+"_"+str(model.hidden_size)+"_"+str(model.direction)+".save"
	except Exception,e:
		print(str(e))
		model_saved_name = "model1_"+str(model.embedding_size)+"_"+str(model.hidden_size)+"_"+str(model.direction)+".save"

	cur_epoch = 0
	trained_sample = 0

	if load_model!="":
		print("Loading model...")
		loaded_data = None
		loaded_data = torch.load(model_saved_name)
		cur_epoch = loaded_data['epoch']
		model.load_state_dict(loaded_data['state_dict'])
		optimizer.load_state_dict(loaded_data['optimizer'])
		trained_sample = loaded_data['trained_sample']
		print(str(trained_sample)+" samples were trained.")
		# print("Loading done!")

	# CE = nn.CrossEntropyLoss()
	NLL = nn.NLLLoss()

	print("Begin training...")
	for epoch in range(epoch_num):
		sample_counter = trained_sample
		total_loss = 0
		total_start_dist = 0
		total_end_dist = 0

		total_start_dist_percent = 0
		total_end_dist_percent = 0
		total_start_order = 0
		total_end_order = 0
		start_time = time.time()

		for sample_i in range(sample_counter,len(train_data)):
			sample = train_data[sample_i]
			sample_counter += 1
			###########################################################
			##Map passage in word embedding
			# passage_em = np.zeros((len(sample.context_token),D))
			# for token_i in range(len(sample.context_token)):
			# 	if sample.context_token[token_i] not in word_em:
			# 		passage_em[token_i] = word_em[UNKNOWNWORD]
			# 	else:
			# 		passage_em[token_i] = word_em[sample.context_token[token_i]]
			# passage_em = torch.from_numpy(np.reshape(passage_em,(len(passage_em),1,D))).float()
			# passage_em = Variable(passage_em)
			###########################################################
			passage_em = model.get_embedding(sample.context_token)
			###########################################################

			###########################################################
			##Map question in word embedding
			# question_em = np.zeros((len(sample.question_token),D))
			# for token_i in range(len(sample.question_token)):
			# 	if sample.question_token[token_i] not in word_em:
			# 		question_em[token_i] = word_em[UNKNOWNWORD]
			# 	else:
			# 		question_em[token_i] = word_em[sample.question_token[token_i]]
			# question_em = torch.from_numpy(np.reshape(question_em,(len(question_em),1,D))).float()
			# question_em = Variable(question_em)
			###########################################################
			question_em = model.get_embedding(sample.question_token)
			###########################################################
			
			true_start = autograd.Variable(torch.LongTensor([sample.start_token]))
			true_end = autograd.Variable(torch.LongTensor([sample.end_token]))

			optimizer.zero_grad()
			my_start,my_end = model(question_em,passage_em)

			###########################################################
			##Some performance statistics
			predict_start_score = my_start.data[0].numpy()
			predict_start = np.argmax(predict_start_score)
			true_start_score = predict_start_score[sample.start_token]
			total_start_order += float(GetOrder(true_start_score,predict_start_score))/len(sample.context_token)
			total_start_dist_percent += float(np.abs(predict_start-sample.start_token))/len(sample.context_token)

			predict_end_score = my_end.data[0].numpy()
			predict_end = np.argmax(predict_end_score)
			true_end_score = predict_end_score[sample.end_token]
			total_end_order += float(GetOrder(true_end_score,predict_end_score))/len(sample.context_token)
			total_end_dist_percent += float(np.abs(predict_end-sample.end_token))/len(sample.context_token)
			###########################################################

			# loss = CE(my_output,true_output)
			loss = NLL(my_start,true_start)+NLL(my_end,true_end)
			total_loss += loss.data[0]

			loss.backward()
			optimizer.step()


			print_every_sample = 100
			if sample_counter%print_every_sample==0:
				print("Loss: "+str(total_loss/print_every_sample))
				total_loss = 0

				print("Average start point distance percent: "+str(float(total_start_dist_percent)/print_every_sample))
				total_start_dist_percent = 0
				print("Average end point distance percent: "+str(float(total_end_dist_percent)/print_every_sample))
				total_end_dist_percent = 0

				print("Average start point order: "+str(float(total_start_order)/print_every_sample))
				total_start_order = 0
				print("Average end point order: "+str(float(total_end_order)/print_every_sample))
				total_end_order = 0

				print("Time: "+str(time.time()-start_time))
				start_time = time.time()

				print("Epoch "+str(epoch)+": "+str(sample_counter)+" samples")

				print("Saving model...")
				save_model({'epoch': epoch,'state_dict': model.state_dict(),
					'optimizer':optimizer.state_dict(),'trained_sample':sample_counter},
					model_saved_name)
				# print("Saving done!")

				print("###########################################################")




if __name__=="__main__":
	D = 50

	###########################################################
	##Read train data from saved file
	print("Reading data...")
	train_data = []
	train_data_file = open("./data/train_data.out")
	qa_object = QA()
	while qa_object.ReadFromFile(train_data_file):
		train_data.append(qa_object)
		qa_object = QA()
	# print(len(train_data))
	# train_data[10000].Show()
	train_data_file.close()
	# print(train_data[10].answer)
	# print(" ".join(train_data[10].context_token[train_data[10].start_token:train_data[10].end_token+1]))

	word_em = ReadWrodEmbedding("./data/processed_word_embedding")
	# print(len(train_data))
	# print(len(word_em))
	# print("Reading finish!")
	###########################################################

	load_model = sys.argv[1]
	if load_model=="true":
		TrainModel(train_data,word_em,D,"load_model")
	else:
		TrainModel(train_data,word_em,D,"")

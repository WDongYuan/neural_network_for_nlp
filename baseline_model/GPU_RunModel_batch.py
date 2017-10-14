from ReadData import *
# from Model import *
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time
import sys
from GPU_ModelBatch import *
###########################################################
#GPU OPTION
###########################################################
import torch.backends.cudnn as cudnn
###########################################################

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
	hidden_size = 500
	embedding_size = D
	epoch_num = 100
	direction = 2
	batch_size = 100
	context_max_length = max([len(sample.context_token) for sample in train_data])
	question_max_length = max([len(sample.question_token) for sample in train_data])


	###########################################################
	#GPU OPTION
	###########################################################
	cudnn.benchmark = True
	###########################################################
	model = ModelBatch(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length)
	optimizer = optim.SGD(model.parameters(), lr=0.01)

	model_saved_name = ""
	try:
		model_saved_name = model.name+"_"+str(model.embedding_size)+"_"+str(model.hidden_size)+"_"+str(model.direction)+".save"
	except:
		# print(str(e))
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
		print(str(cur_epoch)+" epoches were train.")
		print(str(trained_sample)+" samples were trained.")
		cur_epoch += 1
		# print("Loading done!")

	# CE = nn.CrossEntropyLoss()
	NLL = nn.NLLLoss()
	###########################################################
	#GPU OPTION
	###########################################################
	model.cuda()
	###########################################################

	print("Begin training...")
	for epoch in range(epoch_num):
		print("Epoch "+str(epoch))

		cur_epoch += epoch
		sample_counter = trained_sample
		trained_sample = 0

		total_loss = 0.0

		start_order = 0.0
		end_order = 0.0
		start_acc = 0.0
		start_pro = 0.0
		max_pro = 0.0

		start_time = time.time()

		for batch in range(int(len(train_data)/batch_size)):
			sample_counter += batch_size
			batch_obj = [sample for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			batch_context = [sample.context_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			batch_question = [sample.question_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			
			###########################################################
			#GPU OPTION
			###########################################################
			true_start = autograd.Variable(torch.LongTensor([sample.start_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]).cuda(async=True))
			###########################################################
			# true_start = autograd.Variable(torch.LongTensor([sample.start_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]))
			###########################################################

			optimizer.zero_grad()
			my_start,context_length = model(batch_question,batch_context)

			###########################################################
			#GPU OPTION
			###########################################################
			# batch_predict = my_start.data.numpy()
			###########################################################
			batch_predict = my_start.data.cpu().numpy()
			###########################################################
			# print("Batch predict: "+str(batch_predict.shape))
			for i in range(len(batch_predict)):
				sample = batch_obj[i]
				predict_start_score = batch_predict[i][0:context_length[i]]
				# predict_start = np.argmax(predict_start_score)
				true_start_score = predict_start_score[sample.start_token]
				start_pro += true_start_score
				max_pro += np.max(predict_start_score)
				true_order = GetOrder(true_start_score,predict_start_score)
				if true_order==1:
					start_acc += 1
				start_order += float(true_order)/len(sample.context_token)
			start_acc /= batch_size
			start_order /= batch_size
			start_pro /= batch_size
			max_pro /= batch_size


			# predict_end_score = my_end.data[0].numpy()
			# predict_end = np.argmax(predict_end_score)
			# true_end_score = predict_end_score[sample.end_token]
			# total_end_order += float(GetOrder(true_end_score,predict_end_score))/len(sample.context_token)
			# total_end_dist_percent += float(np.abs(predict_end-sample.end_token))/len(sample.context_token)
			###########################################################

			# loss = CE(my_output,true_output)
			# loss = NLL(my_start,true_start)+NLL(my_end,true_end)
			loss = NLL(my_start,true_start)
			total_loss += loss.data[0]

			loss.backward()
			optimizer.step()


			print_every_batch = 1
			if sample_counter%print_every_batch==0:
				print("Loss: "+str(total_loss/print_every_batch))
				total_loss = 0.0

				print("Accuracy: "+str(start_acc))
				start_acc = 0.0

				print("Start point order: "+str(start_order))
				start_order = 0.0

				print("Start point probability: "+str(start_pro))
				start_pro = 0.0

				print("Max probability: "+str(max_pro))
				max_pro = 0.0

				print("Time: "+str(time.time()-start_time))
				start_time = time.time()

				print("Epoch "+str(cur_epoch)+": "+str(sample_counter)+" samples")
				print("###########################################################")

			if sample_counter%10000==0:
				print("Saving model...")
				save_model({'epoch': cur_epoch,'state_dict': model.state_dict(),
					'optimizer':optimizer.state_dict(),'trained_sample':sample_counter},
					model_saved_name)
				print("Saving done!")
				print("###########################################################")
	print("Done!")




if __name__=="__main__":
	D = 50

	###########################################################
	##Read train data from saved file
	print("Reading data...")
	train_data = []
	###########################################################
	#PYTHON VERSION
	###########################################################
	train_data_file = open("./data/train_data.out",encoding='utf-8')
	###########################################################
	# train_data_file = open("./data/train_data.out")
	###########################################################
	qa_object = QA()
	while qa_object.ReadFromFile(train_data_file):
		train_data.append(qa_object)
		qa_object = QA()
	print("Total sample: "+str(len(train_data)))
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

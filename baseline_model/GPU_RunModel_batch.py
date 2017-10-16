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
# from GPU_ModelBatch import *
# from SoftmaxAttention import *
from ConcatLSTM import *
# from TwoPointerModel import *
###########################################################
#GPU OPTION
###########################################################
import torch.backends.cudnn as cudnn
###########################################################
def Accuracy(model,data,pointer_type):
	random.shuffle(data)
	data = data[0:200]
	batch_size = len(data)

	batch_context = [sample.context_token for sample in data]
	batch_question = [sample.question_token for sample in data]
	my_predict,context_length = model(batch_question,batch_context)

	start_order = 0.0
	start_acc = 0.0
	start_pro = 0.0
	max_pro = 0.0

	###########################################################
	#GPU OPTION
	###########################################################
	batch_predict = my_predict.data.cpu().numpy()
	###########################################################
	# batch_predict = my_predict.data.numpy()
	###########################################################
	for i in range(len(batch_predict)):
		sample = data[i]
		predict_start_score = batch_predict[i][0:context_length[i]]

		true_start_score = None
		if pointer_type=="start":
			true_start_score = predict_start_score[sample.start_token]
		elif pointer_type=="end":
			true_start_score = predict_start_score[sample.end_token]

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

	print("Accuracy: "+str(start_acc))
	print("Point order: "+str(start_order))
	print("Point probability: "+str(start_pro))
	print("Max probability: "+str(max_pro))



def GetOrder(val,arr):
	order = 1
	for i in range(len(arr)):
		if arr[i]>val:
			order += 1
	return order
def save_model(state, filename='saved_model.out'):
    torch.save(state, filename)

def TrainModel(train_data,dev_data,word_em,D,load_model,model_mode,learning_rate,hidden_size,pointer_type):

	global UNKNOWNWORD
	# hidden_size = 200
	embedding_size = D
	epoch_num = 100
	direction = 2
	batch_size = 100
	context_max_length = max([len(sample.context_token) for sample in train_data])
	question_max_length = max([len(sample.question_token) for sample in train_data])
	
	max_end_token = max([len(sample.end_token) for sample in train_data])
	print("Max end token: "+str(max_end_token))
	max_answer_length = 10
	# print(context_max_length)
	# print(question_max_length)
	# print(max([len(sample.context_token) for sample in dev_data]))
	# print(max([len(sample.question_token) for sample in dev_data]))


	###########################################################
	#GPU OPTION
	###########################################################
	cudnn.benchmark = True
	###########################################################
	model=None
	# if model_mode=="concat_attention":
	# 	model = ModelBatch(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length)
	# if model_mode=="softmax_attention":
	# 	model = SoftmaxAttentionModel(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length)
	if model_mode=="concat_lstm":
		model = ConcatLSTM(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length)
	# if model_mode=="two_pointer":
	# 	model = TwoPointerModel(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length,max_answer_length)
	if model==None:
		print("No model selected.")
		return
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
	optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

	model_saved_name = ""
	try:
		model_saved_name = model.name+"_"+str(model.embedding_size)+"_"+str(model.hidden_size)+"_"+str(model.direction)+"_"+str(learning_rate)+"_"+pointer_type+".save"
	except:
		# print(str(e))
		model_saved_name = "model1_"+str(model.embedding_size)+"_"+str(model.hidden_size)+"_"+str(model.direction)+"_"+str(learning_rate)+"_"+pointer_type+".save"

	cur_epoch = 0
	trained_sample = 0

	if load_model == "load":
		print("Loading model...")
		loaded_data = None
		loaded_data = torch.load(model_saved_name)
		cur_epoch = loaded_data['epoch']
		model.load_state_dict(loaded_data['state_dict'])
		optimizer.load_state_dict(loaded_data['optimizer'])
		trained_sample = loaded_data['trained_sample']
		print(str(cur_epoch)+" epoches were train.")
		print(str(trained_sample)+" samples were trained.")
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
		# print("Epoch "+str(epoch))

		cur_epoch += 1
		sample_counter = trained_sample
		trained_sample = 0

		# total_loss = 0.0

		# start_order = 0.0
		# end_order = 0.0
		# start_acc = 0.0
		# start_pro = 0.0
		# max_pro = 0.0

		start_time = time.time()

		for batch in range(int(len(train_data)/batch_size)):
			sample_counter += batch_size
			batch_obj = [sample for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			batch_context = [sample.context_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			batch_question = [sample.question_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			# batch_start = [sample.start_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]
			

			###########################################################
			#GPU OPTION
			###########################################################
			true_pos = None
			if pointer_type=="start":
				true_pos = autograd.Variable(torch.LongTensor([sample.start_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]).cuda(async=True))
			elif pointer_type=="end":
				true_pos = autograd.Variable(torch.LongTensor([sample.end_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]).cuda(async=True))
			###########################################################
			# true_pos = None
			# if pointer_type=="start":
			# 	true_pos = autograd.Variable(torch.LongTensor([sample.start_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]))
			# elif pointer_type=="end":
			# 	true_pos = autograd.Variable(torch.LongTensor([sample.end_token for sample in train_data[batch*batch_size:(batch+1)*batch_size]]))
			###########################################################

			optimizer.zero_grad()

			###########################################################
			my_predict,context_length = model(batch_question,batch_context)
			###########################################################
			# my_start,my_end,context_length = model(batch_question,batch_context,batch_start)
			###########################################################

			###########################################################
			loss = NLL(my_predict,true_pos)
			###########################################################
			# total_loss += loss.data[0]

			loss.backward()
			optimizer.step()


			print_every_batch = 10
			if (batch+1)%print_every_batch==0:
				print("Epoch "+str(cur_epoch)+": "+str(sample_counter)+" samples")
				# print(loss.data[0])
				# print("Loss: "+str(total_loss/print_every_batch))
				# total_loss = 0.0
				# print("Accuracy: "+str(start_acc))
				# start_acc = 0.0
				# print("Start point order: "+str(start_order))
				# start_order = 0.0
				# print("Start point probability: "+str(start_pro))
				# start_pro = 0.0
				# print("Max probability: "+str(max_pro))
				# max_pro = 0.0
				print("Time: "+str(time.time()-start_time))
				start_time = time.time()
				print("###########################################################")
			if sample_counter%10000==0:
				print("Dev set performance")
				###########################################################
				Accuracy(model,dev_data,pointer_type)
				###########################################################
				# TwoPointerAccuracy(model,dev_data)
				###########################################################
				print("###########################################################")
			if sample_counter%10000==0:
				print("Saving model...")
				save_model({'epoch': cur_epoch,'state_dict': model.state_dict(),
					'optimizer':optimizer.state_dict(),'trained_sample':sample_counter},
					model_saved_name)
				print("Saving done!")
				print("###########################################################")
	print("Done!")

def Predict(train_data,test_data,word_em,D,learning_rate,hidden_size,model_path,save_path):

	global UNKNOWNWORD
	embedding_size = D
	epoch_num = 100
	direction = 2
	batch_size = 100
	context_max_length = max([len(sample.context_token) for sample in train_data])
	question_max_length = max([len(sample.question_token) for sample in train_data])

	# test_data = test_data[0:1000]

	###########################################################
	#GPU OPTION
	###########################################################
	cudnn.benchmark = True
	###########################################################
	
	model = SoftmaxAttentionModel(embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length)
	# optimizer = optim.SGD(model.parameters(), lr=learning_rate)
	optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

	print("Loading model...")
	loaded_data = None
	loaded_data = torch.load(model_path)
	cur_epoch = loaded_data['epoch']
	model.load_state_dict(loaded_data['state_dict'])
	optimizer.load_state_dict(loaded_data['optimizer'])
	trained_sample = loaded_data['trained_sample']

	###########################################################
	#GPU OPTION
	###########################################################
	model.cuda()
	###########################################################
	print("Predicting...")
	data = test_data
	test_context = [sample.context_token for sample in data]
	test_question = [sample.question_token for sample in data]

	###########################################################
	#PYTHON VERSION
	###########################################################
	predict_file = open(save_path,"w+",encoding='utf-8')
	###########################################################
	# predict_file = open(save_path,"w+")
	###########################################################
	predict_file.write("{")
	is_first = True
	for batch_i in range(int(len(data)/batch_size)+1):
		begin = batch_i*batch_size
		end = (batch_i+1)*batch_size
		if begin>=len(data):
			break
		elif end>len(data):
			end = len(data)
		print("Sample "+str(begin)+" to "+str(end))
		batch_data = data[begin:end]
		batch_context = test_context[begin:end]
		batch_question = test_question[begin:end]

		my_predict,context_length = model(batch_question,batch_context)

		###########################################################
		#GPU OPTION
		###########################################################
		batch_predict = my_predict.data.cpu().numpy()
		###########################################################
		# batch_predict = my_predict.data.numpy()
		###########################################################
		for i in range(len(batch_predict)):
			tmp_score = batch_predict[i][0:context_length[i]]
			predict_idx = np.argmax(tmp_score)
			if not is_first:
				predict_file.write(", ")
			else:
				is_first = False
			predict_file.write("\""+batch_data[i].question_id+"\": \""+" ".join(batch_data[i].context_token[predict_idx:predict_idx+2])+"\"")
			# predict_file.write("\""+batch_data[i].question_id+"\": \""+batch_data[i].context_token[predict_idx]+"\"")
	predict_file.write("}")
	predict_file.close()
		






if __name__=="__main__":
	if len(sys.argv)==1:
		print("python GPU_RunModel_batch.py load softmax_attention 0.1(learning_rate) 200(hidden_size)")
		print("python GPU_RunModel_batch.py not_load concat_attention 0.1(learning_rate) 200(hidden_size)")
		print("python GPU_RunModel_batch.py predict softmax_attention_50_200_2_0.1_start.save dev_predict")
		exit()

	D = 200

	###########################################################
	##Read train data from saved file
	print("Reading data...")
	train_data = []
	dev_data = []
	###########################################################
	#PYTHON VERSION
	###########################################################
	train_data_file = open("./data/train_data.out",encoding='utf-8')
	dev_data_file = open("./data/dev_data.out",encoding='utf-8')
	###########################################################
	# train_data_file = open("./data/train_data.out")
	# dev_data_file = open("./data/dev_data.out")
	###########################################################
	##Read train data
	qa_object = QA()
	while qa_object.ReadFromFile(train_data_file):
		train_data.append(qa_object)
		qa_object = QA()
	print("Total train sample: "+str(len(train_data)))
	train_data_file.close()

	##Read dev data
	qa_object = QA()
	while qa_object.ReadFromFile(dev_data_file):
		dev_data.append(qa_object)
		qa_object = QA()
	print("Total dev sample: "+str(len(dev_data)))
	dev_data_file.close()

	word_em = ReadWrodEmbedding("./data/processed_word_embedding")
	###########################################################

	if sys.argv[1]=="predict":
		model_path = sys.argv[2]
		save_path = sys.argv[3]
		learning_rate = 0.1
		hidden_size = 200
		Predict(train_data,dev_data,word_em,D,learning_rate,hidden_size,model_path,save_path)
		exit()

	load_model = sys.argv[1]
	model_mode = sys.argv[2]
	learning_rate = float(sys.argv[3])
	hidden_size = int(sys.argv[4])
	pointer_type = sys.argv[5]
	TrainModel(train_data,dev_data,word_em,D,load_model,model_mode,learning_rate,hidden_size,pointer_type)


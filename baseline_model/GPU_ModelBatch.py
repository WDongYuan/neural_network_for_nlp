import numpy as np
import json
# from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
###########################################################
#GPU OPTION
###########################################################
import torch.backends.cudnn as cudnn
###########################################################
UNKNOWNWORD = "unknownword"

class ModelBatch(nn.Module):
	def __init__(self,embedding_size,hidden_size,direction,word_em,batch_size,context_max_length,question_max_length):
		super(ModelBatch, self).__init__()
		self.name = "concat_attention"

		self.word_em = word_em
		self.batch_size = batch_size
		self.direction = direction
		self.hidden_size = hidden_size
		self.embedding_size = embedding_size
		self.passage_max_length = context_max_length
		self.question_max_length = question_max_length

		self.q_hidden = self.init_hidden()
		self.q_c_n = self.init_hidden()
		self.p_hidden = self.init_hidden()
		self.p_c_n = self.init_hidden()

		# self.word_embeddings = nn.Embedding(voc_size, embedding_size)
		bidirectional_flag = True if self.direction==2 else False
		self.passage_lstm = nn.LSTM(embedding_size, hidden_size,bidirectional=bidirectional_flag,batch_first=True)
		self.question_lstm = nn.LSTM(embedding_size, hidden_size,bidirectional=bidirectional_flag,batch_first=True)

		# self.start_att_linear1 = nn.Linear(3*2*hidden_size,hidden_size)
		self.start_att_linear1 = nn.Linear(3*self.direction*hidden_size,hidden_size)
		self.start_att_tanh = nn.Tanh()
		self.start_att_linear2 = nn.Linear(hidden_size,1)

		
		# self.end_att_linear1 = nn.Linear(3*self.direction*hidden_size,hidden_size)
		# self.end_att_tanh = nn.Tanh()
		# self.end_att_linear2 = nn.Linear(hidden_size,1)

		self.softmax = nn.Softmax()

	def init_hidden(self):
		###########################################################
		#GPU OPTION
		###########################################################
		return autograd.Variable(torch.rand(self.direction,self.batch_size,self.hidden_size).cuda(async=True))
		###########################################################
		# return autograd.Variable(torch.rand(self.direction,self.batch_size,self.hidden_size))
		###########################################################

	def forward(self,question,passage):
		self.q_hidden = self.init_hidden()
		# print(self.q_hidden)
		self.q_c_n = self.init_hidden()
		self.p_hidden = self.init_hidden()
		self.p_c_n = self.init_hidden()

		question_embedding,question_length = self.get_embedding(question,self.question_max_length)
		passage_embedding,passage_length = self.get_embedding(passage,self.passage_max_length)

		#lstm_out: (seq_len, batch, hidden_size * num_directions)
		#hidden: (num_layers * num_directions, batch, hidden_size)
		#c_n: (num_layers * num_directions, batch, hidden_size)
		q_lstm_out, (self.q_hidden,self.q_c_n) = self.question_lstm(question_embedding,(self.q_hidden,self.q_c_n))
		p_lstm_out, (self.p_hidden,self.p_c_n) = self.passage_lstm(passage_embedding, (self.p_hidden,self.p_c_n))

		q_hidden_list_batch = q_lstm_out
		p_hidden_list_batch = p_lstm_out

		all_question_length = [i*self.question_max_length+question_length[i] for i in range(len(question_length))]
		all_passage_length = [i*self.passage_max_length+passage_length[i] for i in range(len(passage_length))]
		
		tmp_q = q_hidden_list_batch.contiguous().view(-1,self.direction*self.hidden_size)
		tmp_p = p_hidden_list_batch.contiguous().view(-1,self.direction*self.hidden_size)

		###########################################################
		#GPU OPTION
		###########################################################
		question_last_hidden = torch.index_select(tmp_q,0,Variable(torch.LongTensor(question_length).cuda(async=True)))
		passage_last_hidden = torch.index_select(tmp_p,0,Variable(torch.LongTensor(passage_length).cuda(async=True)))
		###########################################################
		# question_last_hidden = torch.index_select(tmp_q,0,Variable(torch.LongTensor(all_question_length)))
		# passage_last_hidden = torch.index_select(tmp_p,0,Variable(torch.LongTensor(all_passage_length)))
		###########################################################

		##Concatanate 2 direction
		q_hidden_new = question_last_hidden.view(self.batch_size,1,self.direction*self.hidden_size).expand(self.batch_size,self.passage_max_length,self.direction*self.hidden_size)
		p_hidden_new = passage_last_hidden.view(self.batch_size,1,self.direction*self.hidden_size).expand(self.batch_size,self.passage_max_length,self.direction*self.hidden_size)

		##Concatanate question,passage and each hidden state in passage
		concat_q_p_h = torch.cat((p_hidden_list_batch,q_hidden_new,p_hidden_new),2)
		start_hidden_score = self.start_att_linear2(self.start_att_tanh(self.start_att_linear1(concat_q_p_h)))
		start_pro = self.softmax(start_hidden_score.view(self.batch_size,-1))

		# end_hidden_score = self.end_att_linear2(self.end_att_tanh(self.end_att_linear1(concat_q_p_h)))
		# end_pro = self.softmax(end_hidden_score.view(self.batch_size,-1))

		return start_pro,passage_length

	def get_embedding(self,sent_token_batch,max_length):
		global UNKNOWNWORD

		# sorted_idx = sorted(range(len(sent_token_batch)),key=lambda k: len(sent_token_batch[k]),reverse=True)
		# sent_token_batch = sorted(sent_token_batch,key=lambda sent_token:len(sent_token),reverse=True)
		sent_length = [len(sent_token) for sent_token in sent_token_batch]

		batch_embedding = np.zeros((self.batch_size,max_length,self.embedding_size))
		for i in range(len(sent_token_batch)):
			sent_token = sent_token_batch[i]
			sentence_em = np.zeros((len(sent_token),self.embedding_size))
			for token_i in range(len(sent_token)):
				if sent_token[token_i] not in self.word_em:
					sentence_em[token_i] = self.word_em[UNKNOWNWORD]
				else:
					sentence_em[token_i] = self.word_em[sent_token[token_i]]
			batch_embedding[i,0:len(sent_token),:] = sentence_em
		###########################################################
		#GPU OPTION
		###########################################################
		batch_embedding = Variable(torch.from_numpy(batch_embedding).float().cuda(async=True))
		###########################################################
		# batch_embedding = Variable(torch.from_numpy(batch_embedding).float())
		###########################################################
		return batch_embedding,sent_length













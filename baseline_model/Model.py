import numpy as np
import json
from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
UNKNOWNWORD = "unknownword"

class Model(nn.Module):
	def __init__(self,embedding_size,hidden_size,direction,word_em):
		super(Model, self).__init__()
		# self.model_name = "model1"

		self.word_em = word_em

		self.direction = direction
		self.hidden_size = hidden_size
		# self.voc_size = voc_size
		self.embedding_size = embedding_size

		# self.word_embeddings = nn.Embedding(voc_size, embedding_size)
		bidirectional_flag = True if self.direction==2 else False
		self.passage_lstm = nn.LSTM(embedding_size, hidden_size,bidirectional=bidirectional_flag)
		self.question_lstm = nn.LSTM(embedding_size, hidden_size,bidirectional=bidirectional_flag)

		# self.start_att_linear1 = nn.Linear(3*2*hidden_size,hidden_size)
		self.start_att_linear1 = nn.Linear(3*self.direction*hidden_size,hidden_size)
		self.start_att_tanh = nn.Tanh()
		self.start_att_linear2 = nn.Linear(hidden_size,1)

		# self.end_att_linear1 = nn.Linear(3*2*hidden_size,hidden_size)
		self.end_att_linear1 = nn.Linear(3*self.direction*hidden_size,hidden_size)
		self.end_att_tanh = nn.Tanh()
		self.end_att_linear2 = nn.Linear(hidden_size,1)

		self.softmax = nn.Softmax()

		self.q_hidden = self.init_hidden()
		self.q_c_n = self.init_hidden()
		self.p_hidden = self.init_hidden()
		self.p_c_n = self.init_hidden()

	def init_hidden(self):
		# return autograd.Variable(torch.zeros(2, 1, self.hidden_size))
		return autograd.Variable(torch.zeros(self.direction, 1, self.hidden_size))

	def forward(self,question,passage):
		self.q_hidden = self.init_hidden()
		self.q_c_n = self.init_hidden()
		self.p_hidden = self.init_hidden()
		self.p_c_n = self.init_hidden()

		question_embedding = question
		passage_embedding = passage

		#lstm_out: (seq_len, batch, hidden_size * num_directions)
		#hidden: (num_layers * num_directions, batch, hidden_size)
		#c_n: (num_layers * num_directions, batch, dden_size)
		q_lstm_out, (self.q_hidden,self.q_c_n) = self.question_lstm(question_embedding.view(len(question_embedding), 1, -1), (self.q_hidden,self.q_c_n))
		p_lstm_out, (self.p_hidden,self.p_c_n) = self.passage_lstm(passage_embedding.view(len(passage_embedding), 1, -1), (self.p_hidden,self.p_c_n))

		# print(self.q_hidden)
		##Concatanate 2 direction
		# q_hidden_new = self.q_hidden.view(1,2*self.hidden_size).expand(len(passage),2*self.hidden_size)
		q_hidden_new = self.q_hidden.view(1,self.direction*self.hidden_size).expand(len(passage),self.direction*self.hidden_size)
		p_hidden_new = self.p_hidden.view(1,self.direction*self.hidden_size).expand(len(passage),self.direction*self.hidden_size)
		# p_hidden_new = self.p_hidden.view(1,2*self.hidden_size).expand(len(passage),2*self.hidden_size)
		p_lstm_out = p_lstm_out.view(len(passage),-1)

		##Concatanate question,passage and each hidden state in passage
		concat_q_p_h = torch.cat((p_lstm_out,q_hidden_new,p_hidden_new),1)
		start_hidden_score = self.start_att_linear2(self.start_att_tanh(self.start_att_linear1(concat_q_p_h))).view(1,-1)
		start_pro = self.softmax(start_hidden_score)

		end_hidden_score = self.end_att_linear2(self.end_att_tanh(self.end_att_linear1(concat_q_p_h))).view(1,-1)
		end_pro = self.softmax(end_hidden_score)

		# result_pro = torch.cat((start_pro,end_pro),1)

		return start_pro,end_pro

	def get_embedding(self,sent_token):
		global UNKNOWNWORD
		sentence_em = np.zeros((len(sent_token),self.embedding_size))
		for token_i in range(len(sent_token)):
			if sent_token[token_i] not in self.word_em:
				sentence_em[token_i] = self.word_em[UNKNOWNWORD]
			else:
				sentence_em[token_i] = self.word_em[sent_token[token_i]]
		sentence_em = torch.from_numpy(np.reshape(sentence_em,(len(sentence_em),1,self.embedding_size))).float()
		sentence_em = Variable(sentence_em)
		return sentence_em













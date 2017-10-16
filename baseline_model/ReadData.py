import numpy as np
import json
# from nltk.tokenize import word_tokenize
# import matplotlib.pyplot as plt
UNKNOWNWORD = "unknownword"
class QA:
	def __init__(self,context=None,question=None,answer=None,answer_start=-1,question_id=None):
		self.token_split_symbol = "##"

		self.context = context
		self.question = question
		self.answer = answer
		self.answer_start = int(answer_start)
		self.question_id = question_id
		if self.context==None:
			return
		###########################################################
		##Combine multiple lines in context,question,answer
		# self.context = " ".join(context.split("\n")).lower()
		# self.question = " ".join(self.question.split("\n")).lower()
		# self.answer = " ".join(self.answer.split("\n")).lower()
		###########################################################
		self.context = " ".join(context.split("\n"))
		self.question = " ".join(self.question.split("\n"))
		self.answer = " ".join(self.answer.split("\n"))
		###########################################################

		###########################################################
		##Tokenization begin
		# self.context_token = word_tokenize(self.context)
		# self.context_token = [token.lower() for token in self.context_token]
		# self.question_token = word_tokenize(self.question)
		# self.question_token = [token.lower() for token in self.question_token]
		# self.answer_token = word_tokenize(self.answer)
		# self.answer_token = [token.lower() for token in self.answer_token]

		# before_answer_length = len("".join(self.context[0:self.answer_start].split(" ")))
		# self.start_token = 0
		# while len("".join(self.context_token[0:self.start_token+1]))<before_answer_length:
		# 	self.start_token += 1
		# self.end_token = self.start_token
		# while "".join(self.answer_token) not in "".join(self.context_token[self.start_token:self.end_token+1]):
		# 	if self.end_token>len(self.context_token):
		# 		self.end_token=len(self.context_token)-1
		# 		print("False")
		# 		return
		# 	self.end_token += 1
		# while "".join(self.answer_token) in "".join(self.context_token[self.start_token+1:self.end_token+1]):
		# 	self.start_token += 1
		# # print(self.start_token)
		# # print(self.end_token)


		# if "".join(self.answer_token) not in "".join(self.context_token[self.start_token:self.end_token+1]):
		# 	print("".join(self.answer_token))
		# 	# print(self.context_token[self.start_token-1:self.start_token+2])
		# 	print("".join(self.context_token[self.start_token:self.end_token+1]))
		# 	print("#######################")
		##Tokenization End
		###########################################################



		# self.context = self.context.encode("utf8")
		# self.question = self.question.encode("utf8")
		# self.answer = self.answer.encode("utf8")
		###########################################################

	def Show(self):
		print("Context: "+self.context)
		print("")
		print("Question: "+self.question)
		print("")
		print("Answer: "+self.answer)
		print("")
		print("Answer Start: "+str(self.answer_start))
		print("")

	def SaveToFile(self,file_token):
		file_token.write(self.context.encode("utf-8")+"\n")
		file_token.write(self.question.encode("utf-8")+"\n")
		file_token.write(self.question_id.encode("utf-8")+"\n")
		file_token.write(self.answer.encode("utf-8")+"\n")
		file_token.write(str(self.answer_start).encode("utf-8")+"\n")
		file_token.write(self.token_split_symbol.join(self.context_token).encode("utf-8")+"\n")
		file_token.write(self.token_split_symbol.join(self.question_token).encode("utf-8")+"\n")
		file_token.write(self.token_split_symbol.join(self.answer_token).encode("utf-8")+"\n")

		file_token.write(str(self.start_token).encode("utf-8")+"\n")
		file_token.write(str(self.end_token).encode("utf-8")+"\n")
		file_token.write(" ".join(self.context_token[self.start_token:self.end_token+1]).encode("utf-8")+"\n")
		file_token.write("\n")

	def ReadFromFile(self,file_token):
		try:
			self.context = file_token.readline().strip()
			self.question = file_token.readline().strip()
			self.question_id = file_token.readline().strip()
			self.answer = file_token.readline().strip()
			self.answer_start = int(file_token.readline().strip())
			self.context_token = file_token.readline().strip().split(self.token_split_symbol)
			self.question_token = file_token.readline().strip().split(self.token_split_symbol)
			self.answer_token = file_token.readline().strip().split(self.token_split_symbol)

			self.start_token = int(file_token.readline().strip())
			self.end_token = int(file_token.readline().strip())
			extract_answer = file_token.readline().strip()
			file_token.readline()
		# except Exception,e:
		except:
			# print(str(e))
			return False
		return True


def ReadWrodEmbedding(path):
	dic = {}
	###########################################################
	#PYTHON VERSION
	###########################################################
	with open(path,encoding='utf-8') as f:
	###########################################################
	# with open(path) as f:
	###########################################################
		for line in f:
			word_em = line.strip().split(" ")
			word = word_em[0]
			em = word_em[1:len(word_em)]
			em = np.array([float(value) for value in em])
			dic[word] = em
	return dic

def ReadTrainData(path="./data/train-v1.1.json"):
	train_data_json = None
	###########################################################
	#PYTHON VERSION
	###########################################################
	with open(path,encoding='utf-8') as f:
	###########################################################
	# with open(path) as f:
	###########################################################
		train_data_json = json.load(f)
	
	# one_article = train_data_json["data"][0]
	# # print(one_article["title"])
	# paragraph = one_article["paragraphs"]
	# print(paragraph)
	qa_data = []
	counter = 0
	for article in train_data_json["data"]:
		for paragraph in article["paragraphs"]:
			for one_qa in paragraph["qas"]:
				counter += 1
				if counter%1000==0:
					print(counter)
				# if counter==1000:
				# 	return qa_data
				qa_data.append(QA(paragraph["context"],one_qa["question"],one_qa["answers"][0]["text"],int(one_qa["answers"][0]["answer_start"]),one_qa["id"]))
	return qa_data

def CreateUnknownWord(qa_list,word_em,D,save_path="./data/processed_word_embedding"):
	global UNKNOWNWORD
	count_dic = {}
	for word,em in word_em.items():
		count_dic[word] = 0

	# unknown = {}
	for qa in qa_list:
		for token in qa.context_token:
			if token in count_dic:
				count_dic[token] += 1

	unknown_word_count = 0
	unknown_em = np.zeros((D,))
	for word,count in count_dic.items():
		if count == 0:
			unknown_em += word_em[word]
			unknown_word_count += 1
			del word_em[word]
	unknown_em /= unknown_word_count
	word_em[UNKNOWNWORD] = unknown_em
	print(len(word_em))

	file = open(save_path,"w+")
	for word,em in word_em.items():
		tmp_str = word
		for i in range(len(em)):
			tmp_str += (" "+str(em[i]))
		file.write(tmp_str+"\n")
	file.close()





if __name__=="__main__":
	D = 50
	# word_em = ReadWrodEmbedding("./data/glove.6B/glove.6B."+str(D)+"d.txt")
	# CreateUnknownWord(train_data,word_em,D,"./data/processed_word_embedding")
	###########################################################
	##Read train data from json file and then save it.
	# tmp_counter = 0
	# train_data = ReadTrainData("./data/train-v1.1.json")
	# save_train_data_file = open("./data/train_data.out","w+")
	# for qa in train_data:
	# 	tmp_counter += 1
	# 	qa.SaveToFile(save_train_data_file)
	# 	if tmp_counter%10000==0:
	# 		print(tmp_counter)
	# save_train_data_file.close()
	# print("Finish reading...")
	###########################################################

	###########################################################
	##Check context line
	# tmp_counter = 0
	# for qa in train_data:
	# 	if len(qa.context.split("\n"))>1:
	# 		print(qa.context)
	# 		break
	# 		tmp_counter += 1
	# print(tmp_counter)
	###########################################################

	###########################################################
	#Read train data from saved file
	# train_data = []
	# train_data_file = open("./data/train_data.out")
	# qa_object = QA()
	# while qa_object.ReadFromFile(train_data_file):
	# 	train_data.append(qa_object)
	# 	qa_object = QA()
	# # print(len(train_data))
	# # train_data[10000].Show()
	# train_data_file.close()
	# print(len(train_data))

	# word_em = ReadWrodEmbedding("./data/processed_word_embedding")
	# print(len(train_data))
	# print(len(word_em))
	###########################################################









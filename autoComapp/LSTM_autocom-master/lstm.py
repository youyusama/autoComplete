import sys
import torch
import pdb
import random
import string
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import time
import math
import copy

class Utils():
    @staticmethod
    def time_since(since):
        s = time.time() - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)


class Inferencer():
    def __init__(self, model_file, voc_file, device):
        self.model_file = model_file
        self.device = device

        # load a trained model
        self.lstm = torch.load(self.model_file).to(self.device)
        #self.vocabularyLoader = VocabularyLoader("cre-utf8.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("docker.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("caiding.txt", self.device)
        #self.vocabularyLoader = VocabularyLoader("suanfa.txt", self.device)
        self.vocabularyLoader = VocabularyLoader(voc_file, self.device)


    def getText(self, prime_str, predict_len, temperature=0.8):
        prime_input = self.vocabularyLoader.char_tensor(prime_str)
        # print(prime_input)
        hidden = self.lstm.init_hidden().to(self.device)
        # print(hidden)
        cell_state = self.lstm.init_cell_state().to(self.device)
        # print(cell_state)
        for p in range(len(prime_str) - 1):
            output, hidden, cell_state = self.lstm(prime_input[p], hidden, cell_state)
            # print(output)

        inp = prime_input[-1]
        # print(inp)
        sentence = prime_str
        for i in range(predict_len):
            output, hidden, cell_state = self.lstm(inp, hidden, cell_state)
            output_dist = output.div(temperature).exp()
            # print(output_dist)

            #only use the first element in the array  
            predict = torch.multinomial(output_dist, num_samples=1)[0]
            # print(predict.item())
            #predict = torch.multinomial(output_dist, num_samples=1)#it's wrong

            # 0 means the only one element in the string
            try:
                predict_char = self.vocabularyLoader.index2char[predict.item()]
            except Exception as e:
                # pdb.set_trace()
                print(e)
            inp = predict
            # print(predict_char)
            sentence += predict_char

        return sentence

class LSTM(nn.Module):
    def __init__(self, voc_size, embedding_dim, hidden_size):
        super(LSTM, self).__init__()
        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(voc_size, embedding_dim)
        #forget gate
        self.wf = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uf = nn.Linear(hidden_size, hidden_size, bias = True)
        #input gate
        self.wi = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.ui = nn.Linear(hidden_size, hidden_size, bias =True)
        #ouput gate
        self.wo = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uo = nn.Linear(hidden_size, hidden_size, bias = True)
        #for updating cell state vector
        self.wc = nn.Linear(embedding_dim, hidden_size, bias = False)
        self.uc = nn.Linear(hidden_size, hidden_size, bias = True)
        #gate's activation function
        self.sigmoid = nn.Sigmoid()
        #activation function on the updated cell state 
        self.tanh = nn.Tanh()
        #distribution of the prediction
        self.out = nn.Linear(hidden_size, voc_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, input, hidden, cell_state):
        # print(input.size())
        # print(hidden.size())
        embed_input = self.embedding(input)
        #forget gate's activation vector
        f = self.sigmoid(self.wf(embed_input) + self.uf(hidden))
        # print(self.wf(embed_input).size())
        # print(self.uf(hidden).size())
        #input gate's activation vector
        i = self.sigmoid(self.wi(embed_input) + self.ui(hidden))
        #output gate's activation vector
        o = self.sigmoid(self.wo(embed_input) + self.uo(hidden))
        tmp = self.tanh(self.wc(embed_input) + self.uc(hidden))
        updated_cell_state = torch.mul(cell_state, f) + torch.mul(i, tmp)  
        updated_hidden = torch.mul(self.tanh(updated_cell_state), o)
        output = self.softmax(self.out(updated_hidden))
        return output, updated_hidden, updated_cell_state
         
    def init_hidden(self): 
        return Variable(torch.zeros(self.hidden_size))

    def init_cell_state(self):
        return Variable(torch.zeros(self.hidden_size))


class VocabularyLoader():
    def __init__(self, filename, device):
        self.character_table = {} 
        self.index2char = {}
        self.n_chars = 0
        self.device = device
        with open(filename,'r',encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                for w in line:
                    if w not in self.character_table:
                        self.character_table[w] = self.n_chars 
                        self.index2char[self.n_chars] = w
                        self.n_chars += 1
        #print(self.n_chars)
                    

    # Turn string into list of longs
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.character_table[string[c]]
            except Exception as e:
                #pdb.set_trace()
                print(string[c])
                print(e)
        return Variable(tensor).to(self.device)


class DataLoader():
    def __init__(self, filename, chunk_len, device):
        with open(filename,'r') as f:
            lines=f.readlines()
        self.content = "".join(lines)
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader(filename, self.device)


    def next_chunk(self): 
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + chunk_len 
        if(end_index > self.file_len):
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])


class objlstm:
    def doautocom(self,con):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        words = con
        voc_file = r'D:\CNM\NTeat\autoCom\autoComapp\LSTM_autocom-master\EN-ATP-V226.txt'
        inferencer = Inferencer(r'D:\CNM\NTeat\autoCom\autoComapp\LSTM_autocom-master\lstm_99_99.model', voc_file, device)
        re=[]
        for i in range(3):
            sentence = inferencer.getText(words,predict_len=40)
            re.append(sentence.splitlines()[0])

        return re
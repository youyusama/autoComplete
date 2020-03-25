import sys
import torch
import pdb
import random
import string
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, model_file, voc_file, n_gram, device):
        self.model_file = model_file
        self.device = device
        self.n_gram = n_gram

        # load a trained model
        self.ngram = torch.load(self.model_file).to(self.device)
        self.vocabularyLoader = VocabularyLoader(voc_file, self.device)


    def getText(self, prime_str, predict_len, temperature=0.8):      
        prime_input = self.vocabularyLoader.char_tensor(prime_str.split())
        # hidden = self.lstm.init_hidden().to(self.device)
        # cell_state = self.lstm.init_cell_state().to(self.device)
        # for p in range(len(prime_str) - 1):
        #     output, hidden, cell_state = self.lstm(prime_input[p], hidden, cell_state)
        # output = self.ngram(prime_input)
        # inp = prime_input[-1]
        sentence = prime_str
        print(sentence)
        for i in range(predict_len):
            # print(prime_input[-n_gram:])
            output, embeds = self.ngram(prime_input[1-self.n_gram:])
            output_dist = output.div(temperature).exp()

            #only use the first element in the array  
            predict = torch.multinomial(output_dist, num_samples=1)[0]
            # print(predict)

            # 0 means the only one element in the string
            try:
                predict_char = self.vocabularyLoader.index2char[predict.item()]
            except Exception as e:
                # pdb.set_trace()
                print(e)
            inp = predict
            prime_input = torch.cat((prime_input, torch.tensor(predict)))
            sentence += " "
            sentence += predict_char 

        return sentence


class Trainer():
    def __init__(self, voc_size, n_gram, device):
        self.embedding_dim = 128
        self.hidden_size = 128
        self.n_gram = n_gram
        self.learning_rate = 1e-3
        self.n_step = 500000
        self.voc_size = voc_size
        self.device = device


    def __create_model(self):
        ngram = NGram(self.voc_size, self.embedding_dim, self.hidden_size, self.n_gram)
        return ngram


    def train_within_step(self, ngram, optimizer, criterion, inp, target, chunk_len):
        ngram.zero_grad()
        loss = 0
        
        for c in range(chunk_len-n_gram):
            try:
                log_probs, embeds = ngram(inp[c:c+n_gram-1])
                loss += criterion(log_probs, target[c].unsqueeze(0))
            except Exception as e:
                # pdb.set_trace()
                print(e)

        loss.backward()
        optimizer.step()

        return loss.data.item() / (chunk_len-1)


    def train_model(self, dataloader):
        ngram = self.__create_model()
        ngram = ngram.to(self.device)
        optimizer = torch.optim.Adam(ngram.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()

        start = time.time()
        for step in range(1, self.n_step+1):
            input, target = dataloader.next_chunk()
            loss = self.train_within_step(ngram, optimizer, criterion, input, target, dataloader.chunk_len)
            print('[%s (%d %d%%) %.4f]' % (Utils.time_since(start), step, step / self.n_step* 100, loss))

            if step % 2000 == 0:
                torch.save(ngram, "ngram_"+"{}".format(step)+".model")

class NGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_gram):
        super(NGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear((n_gram-1) * embedding_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, self.embeddings



class VocabularyLoader():
    def __init__(self, filename, device):
        self.character_table = {} 
        self.index2char = {}
        self.n_chars = 0
        self.device = device
        with open(filename,'r') as f:
            lines=f.readlines()
            for line in lines:
                for w in line.split():
                    if w not in self.character_table:
                        self.character_table[w] = self.n_chars 
                        self.index2char[self.n_chars] = w
                        self.n_chars += 1
                    

    # Turn string into list of longs
    def char_tensor(self, string):
        # string = string.split()
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.character_table[string[c]]
            except Exception as e:
                #pdb.set_trace()
                print(e)
        return Variable(tensor).to(self.device)


class DataLoader():
    def __init__(self, filename, chunk_len, n_gram, device):
        with open(filename,'r') as f:
            lines=f.readlines()
        self.content = " ".join(lines).split()
        self.file_len = len(self.content)
        self.chunk_len = chunk_len
        self.n_gram = n_gram
        self.device = device
        self.vocabularyLoader = VocabularyLoader(filename, self.device)


    def next_chunk(self): 
        chunk = self.__random_chunk()
        input = chunk[:-1]
        target = chunk[n_gram-1:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len 
        if(end_index > self.file_len):
            return self.vocabularyLoader.char_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.char_tensor(self.content[start_index:end_index])


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sys.argv.append('train')
    sys.argv.append('EN-ATP-V226.txt')
    sys.argv.append(5) # n_gram的n,表示预测当前token会参考之前的n-1个token

    # sys.argv.append('inference')
    # sys.argv.append('This document defines the ATP software requirements assigned from the onboard train control subsystem')
    # sys.argv.append('EN-ATP-V226.txt')
    # sys.argv.append(10)

    if(len(sys.argv)<2):
        print("usage: ngram [train file | inference (words vocabfile) ]")
        print("e.g. 1: ngram train cre-utf8.txt")
        print("e.g. 2: ngram inference words cre-utf8.txt")
        sys.exit(0) 
    method = sys.argv[1]

    if(method == "train"):
        filename = sys.argv[2]
        n_gram = sys.argv[3]
        chunk_len = 100
        dataloader = DataLoader(filename,chunk_len, n_gram, device)
        trainer = Trainer(dataloader.vocabularyLoader.n_chars, n_gram, device)
        trainer.train_model(dataloader)
    elif(method == "inference"):
        words = sys.argv[2]
        voc_file = sys.argv[3]
        n_gram = sys.argv[4]
        inferencer = Inferencer("ngram_10_6000.model", voc_file, n_gram, device)
        sentence = inferencer.getText(words, predict_len=100)
        print(sentence)
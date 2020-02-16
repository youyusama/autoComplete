import torch 
from torch.autograd import Variable
import json


class VocabularyLoader_char():
    def __init__(self, filename, device):
        self.character_table = {}
        self.index2char = {}
        self.n_chars = 0
        self.device = device
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            for line in lines:
                for w in line:
                    if w not in self.character_table:
                        self.character_table[w] = self.n_chars
                        self.index2char[self.n_chars] = w
                        self.n_chars += 1
        # print(self.n_chars)

    # Turn string into list of longs
    def char_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.character_table[string[c]]
            except Exception as e:
                # pdb.set_trace()
                print(string[c])
                print(e)
        return Variable(tensor).to(self.device)


class VocabularyLoader_token():
    def __init__(self, filename, device):
        self.token_table = {}
        self.index2token = {}
        self.n_tokens = 0
        self.device = device
        f = open(filename, 'r', encoding='UTF-8')
        for lines in f:
            ls = lines.replace('\n', ' ').replace('\t', ' ')
            token_lists = ls.split(' ')
            token_lists = [i for i in token_lists if (len(str(i))) != 0]
            for i in token_lists:
                if i is not '':
                    if i not in self.token_table:
                        self.token_table[i] = self.n_tokens
                        self.index2token[self.n_tokens] = i
                        self.n_tokens += 1
        self.token_table['UNKNOWN'] = self.n_tokens
        self.index2token[self.n_tokens] = 'UNKNOWN'
        self.n_tokens+=1
        # print(self.n_tokens)

    # Turn tokens into list of longs
    def token_tensor(self, tokens):
        tensor = torch.zeros(len(tokens)).long()
        for c in range(len(tokens)):
            try:
                tensor[c] = self.token_table[tokens[c]]
            except Exception as e:
                # pdb.set_trace()
                print(tokens[c])
                print(e)
        return Variable(tensor).to(self.device)


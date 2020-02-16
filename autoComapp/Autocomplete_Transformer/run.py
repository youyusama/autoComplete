import torch
import sys
import seaborn
from torch.autograd import Variable
import numpy as np
import random
from Dataloader import DataLoader_token


def subsequent_mask(size):
    "屏蔽后续位置"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def choose_options(model, memory, src, src_mask, ys):
    out = model.decode(memory, src_mask, Variable(ys[1]), Variable(subsequent_mask(ys[1].size(1)).type_as(src.data)))
    prob = model.generator(out[:, -1])
    dict = {}
    for j in range(prob.size()[-1]):
        dict[j] = prob[0][j].item()
    sort_dict = sorted(zip(dict.values(), dict.keys()), reverse=True)
    options = sort_dict[:beam_search_number]
    result=[]
    for i in range(beam_search_number):
        result.append([ys[0]+options[i][0],torch.cat([ys[1], torch.ones(1, 1).type_as(src.data).fill_(options[i][1])], dim=1)])
    return result


def beam_search_decode(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    ys = [0,torch.ones(1, 1).fill_(src[0][-1].cpu().numpy().item()).type_as(src.data)]
    reserved_options = choose_options(model, memory, src, src_mask, ys)
    for i in range(max_len-1):
        tmp_options=[]
        for j in range(len(reserved_options)):
            tmp_options+=choose_options(model,memory,src,src_mask,reserved_options[j])
        tmp_options=sorted(tmp_options,reverse=True)[:beam_search_number]
        reserved_options=tmp_options
    return reserved_options


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


class DataLoader_token():
    def __init__(self, filename, chunk_len, device):
        with open(filename, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        self.content = "".join(lines)
        self.token_list = self.content.replace('\n', ' ').replace('\t', ' ').split(' ')
        self.token_list = [i for i in self.token_list if (len(str(i))) != 0]
        self.file_len = len(self.token_list)
        self.chunk_len = chunk_len
        self.device = device
        self.vocabularyLoader = VocabularyLoader_token(filename, self.device)

    def next_chunk(self):
        chunk = self.__random_chunk()
        input = chunk[:-9]
        target = chunk[20:]
        return input, target

    def __random_chunk(self):
        start_index = random.randint(0, self.file_len-self.chunk_len)
        end_index = start_index + self.chunk_len
        if end_index > self.file_len:
            return self.vocabularyLoader.token_tensor(self.__random_chunk())
        else:
            return self.vocabularyLoader.token_tensor(self.token_list[start_index:end_index])


seaborn.set_context(context="talk")

chunk_len=30   #每次训练用多少个字符
predict_length=5   #预测结果的长度（字符数）
beam_search_number=3 #束搜索的大小

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sys.argv.append('EN-ATP-V226.txt')
    sys.argv.append('transformer_overlap_11_10_1500.model')
    # 输入的句子
    sys.argv.append('ATP')

    filename = sys.argv[1]
    trained_model_name = sys.argv[2]
    words = sys.argv[3]

    model = torch.load(trained_model_name).cuda()
    model.eval()

    dataloader = DataLoader_token(filename, chunk_len, device)
    word_list = words.replace('\n', ' ').replace('\t', ' ').split(' ')
    word_list = [i for i in word_list if (len(str(i))) != 0]
    src = Variable(dataloader.vocabularyLoader.token_tensor(word_list).unsqueeze(0))
    src_mask = Variable((src != 0).unsqueeze(-2))

    output_embed_list = beam_search_decode(model, src, src_mask, max_len=predict_length)

    print("Inputs: ", words)

    for j in range(len(output_embed_list)):
        output_embed = output_embed_list[j][1][0].cpu().numpy()
        result = []
        for i in output_embed:
            result.append(dataloader.vocabularyLoader.index2token[i])
        result = result[1:]
        result = " ".join(result)
        print(result)

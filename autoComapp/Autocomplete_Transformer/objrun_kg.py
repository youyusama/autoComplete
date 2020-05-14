import torch
import sys
sys.path.append(r'D:\CNM\NTeat\autoCom\autoComapp\Autocomplete_Transformer')
import seaborn
import json
from torch.autograd import Variable
from Batch import Batch, Batch_kg
from Optim import NoamOpt, LabelSmoothing
from Model import make_model, make_model_kg
from Train import run_epoch, greedy_decode, beam_search_decode, SimpleLossCompute, run_epoch_kg, beam_search_decode_kg
from Dataloader import DataLoader_char, DataLoader_token, DataLoader_token_kg
from HyperParameter import chunk_len, batch, nbatches, transformer_size, epoch_number, epoches_of_loss_record, \
    predict_length



class objrun_kg:
    ents = []
    with open(r'D:\CNM\NTeat\autoCom\autoComapp\Autocomplete_Transformer\kg_embed\entity2id.txt') as fin:
        fin.readline()
        for line in fin:
            name, id = line.strip().split("\t")
            ents.append(name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    filename = r'D:\CNM\NTeat\autoCom\autoComapp\models\EN-ATP-V226.txt'
    trained_model_name = r'D:\CNM\NTeat\autoCom\autoComapp\models\transformer_kg_1500_20_9_3.model'
    model = torch.load(trained_model_name, map_location='cpu')
    model.eval()
    dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)


    def __init__(self):
        seaborn.set_context(context="talk")

        # cre文本匹配
        # 输入数据处理
        # 共有nbatches*batch*(chunklen-1)条数据
        def data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device):
            "为src-ent-tgt复制任务生成随机数据"
            for i in range(nbatches):
                data_src = torch.empty(1, chunk_len - 1).long().to(device)
                data_ent = torch.empty(1, chunk_len - 1).long().to(device)
                data_tgt = torch.empty(1, 2).long().to(device)
                for k in range(batch):
                    src_tgt_pair = dataloader.next_chunk()
                    for j in range(0, len(src_tgt_pair)):
                        data_src = torch.cat([data_src, src_tgt_pair[j][0].unsqueeze(0)])
                        data_ent = torch.cat([data_ent, src_tgt_pair[j][1].unsqueeze(0)])
                        data_tgt = torch.cat([data_tgt, src_tgt_pair[j][2].unsqueeze(0)])
                    data_src = data_src[1:]
                    data_ent = data_ent[1:]
                    data_tgt = data_tgt[1:]
                src = Variable(data_src, requires_grad=False)
                ent = Variable(data_ent, requires_grad=False)
                tgt = Variable(data_tgt, requires_grad=False)
                yield Batch_kg(src, ent, tgt, -1)

        def data_gen_overlap_kg(dataloader, batch, nbatches):
            "为src-ent-tgt复制任务生成随机数据"
            for i in range(nbatches):
                data_src = dataloader.next_chunk()[0].unsqueeze(0)
                data_ent = dataloader.next_chunk()[1].unsqueeze(0)
                data_tgt = dataloader.next_chunk()[2].unsqueeze(0)
                for j in range(batch - 1):
                    data_src = torch.cat([data_src, dataloader.next_chunk()[0].unsqueeze(0)], 0)
                    data_ent = torch.cat([data_ent, dataloader.next_chunk()[1].unsqueeze(0)], 0)
                    data_tgt = torch.cat([data_tgt, dataloader.next_chunk()[2].unsqueeze(0)], 0)

                src = Variable(data_src, requires_grad=False)
                ent = Variable(data_ent, requires_grad=False)
                tgt = Variable(data_tgt, requires_grad=False)
                yield Batch_kg(src, ent, tgt, 0)


    def doautocom(self,con):
        word_list = con.replace('\n', ' ').replace('\t', ' ').split(' ')
        word_list = [i for i in word_list if (len(str(i))) != 0]
        src = Variable(self.dataloader.vocabularyLoader.token_tensor(word_list).unsqueeze(0))
        src_mask = Variable((src != 0).unsqueeze(-2))
        ent = Variable(torch.Tensor([24] * len(word_list)).long()).to(self.device)
        ents_list = []
        for i in range(len(self.dataloader.kg)):
            if con.find(" " + self.dataloader.kg[i] + " ") != -1:
                ents_list.append(self.dataloader.kg[i])
        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()
            if word_list.index(key[0]) >= 0:
                ent[word_list.index(key[0])] = self.dataloader.kg.index(" ".join(key))
        ent = ent.unsqueeze(0)

        ent_mask = None
        # print(src)
        # print(ent)
        re=[]
        output_embed_list = beam_search_decode_kg(self.model, src, src_mask, ent, ent_mask, max_len=predict_length)
        for j in range(len(output_embed_list)):
            output_embed = output_embed_list[j][1][0].cpu().numpy()
            result = []
            for i in output_embed:
                result.append(self.dataloader.vocabularyLoader.index2token[i])
            # print("result: ", result)
            result = result[1:]
            result = " ".join(result)
            #print(result)
            re.append(result)
        return re
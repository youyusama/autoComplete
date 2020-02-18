import torch
import sys
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


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sys.argv.append('train')
    sys.argv.append('inference')
    sys.argv.append('EN-ATP-V226.txt')
    sys.argv.append('transformer_kg_1000_10_9_3.model')
    sys.argv.append('According to the Location Reports of the CC  ,  the ZC calculates ')

    if len(sys.argv) < 2:
        print("usage: lstm [train file | inference (words vocabfile) ]")
        print("e.g. 1: lstm train cre-utf8.txt")
        print("e.g. 2: lstm inference cre-utf8.txt words")
        sys.exit(0)

    method = sys.argv[1]

    ents = []
    with open("kg_embed/entity2id.txt") as fin:
        fin.readline()
        for line in fin:
            name, id = line.strip().split("\t")
            ents.append(name)

    if method == "train":
        filename = sys.argv[2]
        dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
        V = dataloader.vocabularyLoader.n_tokens  # vocabolary size

        criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
        criterion.cuda()
        model = make_model_kg(V, V, "kg_embed/embedding.vec.json", N=transformer_size)
        # model.cuda()
        model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        for epoch in range(epoch_number):
            if epoch % epoches_of_loss_record == 0:
                f = open("procedure.txt", "a+")
                f.write("step:%d \n" % epoch)
                f.close()
            print("step: ", epoch)
            model.train()
            # run_epoch_kg("train", data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device), model,
            #           SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
            run_epoch_kg("train", data_gen_overlap_kg(dataloader, batch, nbatches), model,
                         SimpleLossCompute(model.generator, criterion, model_opt), nbatches, epoch)
            model.eval()
            # run_epoch_kg("test ", data_gen_token_kg(dataloader, batch, nbatches, chunk_len, device), model,
            #           SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)
            run_epoch_kg("test ", data_gen_overlap_kg(dataloader, batch, nbatches), model,
                         SimpleLossCompute(model.generator, criterion, None), nbatches, epoch)

    elif method == "inference":

        print('Input:', sys.argv[4])
        print('Options: ')

        filename = sys.argv[2]
        trained_model_name = sys.argv[3]
        words = sys.argv[4]
        model = torch.load(trained_model_name, map_location='cpu')
        model.eval()
        dataloader = DataLoader_token_kg(filename, ents, chunk_len, device)
        word_list = words.replace('\n', ' ').replace('\t', ' ').split(' ')
        word_list = [i for i in word_list if (len(str(i))) != 0]
        src = Variable(dataloader.vocabularyLoader.token_tensor(word_list).unsqueeze(0))
        src_mask = Variable((src != 0).unsqueeze(-2))
        ent = Variable(torch.Tensor([24] * len(word_list)).long()).to(device)
        ents_list = []
        for i in range(len(dataloader.kg)):
            if words.find(" " + dataloader.kg[i] + " ") != -1:
                ents_list.append(dataloader.kg[i])
        for i in range(len(ents_list)):
            key = ents_list[i].strip().split()
            if word_list.index(key[0]) >= 0:
                ent[word_list.index(key[0])] = dataloader.kg.index(" ".join(key))
        ent = ent.unsqueeze(0)

        ent_mask = None
        # print(src)
        # print(ent)
        output_embed_list = beam_search_decode_kg(model, src, src_mask, ent, ent_mask, max_len=predict_length)
        for j in range(len(output_embed_list)):
            output_embed = output_embed_list[j][1][0].cpu().numpy()
            result = []
            for i in output_embed:
                result.append(dataloader.vocabularyLoader.index2token[i])
            # print("result: ", result)
            result = result[1:]
            result = " ".join(result)
            print(result)
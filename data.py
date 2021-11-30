import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

import codecs

class NMTDataset(Dataset):
    def __init__(self, data_dir, sp, mode, src='standard_form',max_len=500):
        super(NMTDataset,self).__init__()

        if src == 'standard_form':
            tgt = 'dialect_form'
        elif src == 'dialect_form':
            tgt = 'standard_form'
        else:
            raise NotImplementedError
        
        self.df = pd.read_csv(f"{data_dir}/{mode}.csv", encoding='utf-8')
        src_list = self.df[src].values.tolist()
        tgt_list = self.df[tgt].values.tolist()

        self.sp = sp
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()
        self.max_len = max_len

        self.src_data = self.get_tokenized_docs(src_list)
        self.tgt_data = self.get_tokenized_docs(tgt_list)

        self.token_to_idx,self.idx_to_token = self.get_vocab()
        
    def get_vocab(self):
        token_to_idx = {}
        idx_to_token = {}
        for id_ in range(self.sp.get_piece_size()):
            token = self.sp.id_to_piece(id_)
            idx_to_token[id_] = token
            token_to_idx[token] = id_
        return token_to_idx,idx_to_token

    def get_tokenized_docs(self,text_list):
        tokenized_docs = []
        for txt in text_list:
            seq = self.sp.encode_as_ids(txt)
            seq = torch.cat((
                torch.tensor([self.bos_id]),
                torch.tensor(seq),
                torch.tensor([self.eos_id])
            ))
            # seq = [self.bos_id] + seq + [self.eos_id]
            # if len(seq)> self.max_len:
            #     import ipdb;ipdb.set_trace()
            # rest = self.max_len - len(seq)
            # seq = torch.tensor(seq + [self.pad_id]*rest)

            tokenized_docs.append(seq)
        return tokenized_docs
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self,idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]

        return src, tgt

def collate_fn(batch):
    src_batch = [src for src,tgt in batch]
    tgt_batch = [tgt for src,tgt in batch]
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=3)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=3)

    return src_batch, tgt_batch

if __name__ == '__main__':
    # Check dataloader
    data_dir = '/nas/datahub/kr-dialect'
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{data_dir}/bpe.model')
    
    dataset = NMTDataset(data_dir,sp,mode='train')
    loader = DataLoader(dataset,batch_size=2,shuffle=True,collate_fn=collate_fn)
    src,tgt = next(iter(loader))
    import ipdb;ipdb.set_trace()
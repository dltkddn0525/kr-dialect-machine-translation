import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm

import codecs

class NMTDataset(Dataset):
    def __init__(self, data_dir, sp, mode, src='dialect_form',use_loc=False):
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
        self.location_label = self.df['label'].values.tolist()

        self.sp = sp
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

        self.src_data = self.get_tokenized_docs(src_list,use_loc=use_loc)
        self.tgt_data = self.get_tokenized_docs(tgt_list,use_loc=False)

        self.token_to_idx,self.idx_to_token = self.get_vocab()
        
        
    def get_vocab(self):
        token_to_idx = {}
        idx_to_token = {}
        for id_ in range(self.sp.get_piece_size()):
            token = self.sp.id_to_piece(id_)
            idx_to_token[id_] = token
            token_to_idx[token] = id_
        return token_to_idx,idx_to_token

    def get_tokenized_docs(self,text_list,use_loc=False):
        tokenized_docs = []
        for i in range(len(text_list)):
            txt = text_list[i]
            seq = self.sp.encode_as_ids(txt)
            label_token = self.location_label[i]+self.sp.get_piece_size()
            if use_loc:
                seq = torch.cat((
                    torch.tensor([self.bos_id]),
                    torch.tensor([label_token]),
                    torch.tensor(seq),
                    torch.tensor([self.eos_id])
                ))
            else:
                seq = torch.cat((
                    torch.tensor([self.bos_id]),
                    torch.tensor(seq),
                    torch.tensor([self.eos_id])
                ))
            tokenized_docs.append(seq)
        return tokenized_docs
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self,idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]
        label = self.location_label[idx]

        return src, tgt, torch.tensor(label)

def collate_fn(batch):
    src_batch = [src for src,tgt,label in batch]
    tgt_batch = [tgt for src,tgt,label in batch]
    label_batch = [label for src,tgt,label in batch]
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=3)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=3)

    return src_batch, tgt_batch, torch.tensor(label_batch)

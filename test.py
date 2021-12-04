import argparse
import os
import torch
import json
import sentencepiece as spm
from nltk.translate.bleu_score import corpus_bleu

from data import NMTDataset, collate_fn
from model import Transformer, Encoder, Decoder
from utils import AverageMeter, Logger

parser = argparse.ArgumentParser(description='Transformer dialect machine translation')
parser.add_argument('--data-dir', default='/nas/datahub/kr-dialect',type=str,
                    help='path to data of specific domain')
parser.add_argument('--ckpt', default='./trial1/last_model.pth',type=str,
                    help='Save path')
args = parser.parse_args()

def main():

    # Get configuration
    config_path = os.path.join(os.path.dirname(args.ckpt),'configuration.json')
    with open(config_path,'r') as f:
        config = json.load(f)

    # Load Dataset
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{args.data_dir}/bpe_{config['vocab_size']}.model')
    val_dataset = NMTDataset(args.data_dir,sp,'val')
    test_dataset = NMTDataset(args.data_dir,sp,'test')

    # Load Model
    device = torch.device('cuda:0')
    num_vocab = sp.get_piece_size()


    encoder = Encoder(
        input_dim=num_vocab, 
        hidden_dim=config['hidden_dim'], 
        n_layers=config['enc_layer'], 
        n_heads=config['enc_head'], 
        pf_dim=512, 
        dropout_ratio=0.1, 
        device=device)

    decoder = Decoder(
        output_dim=num_vocab, 
        hidden_dim=config['hidden_dim'], 
        n_layers=config['dec_layer'], 
        n_heads=config['dec_head'], 
        pf_dim=512, 
        dropout_ratio=0.1, 
        device=device)
    model = Transformer(
        encoder,
        decoder,
        sp.pad_id(),
        sp.pad_id(),
        device
    ).to(device)

    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt)

    val_perf = evaluate(model,val_dataset,sp,device)
    test_perf = evaluate(model,test_dataset,sp,device)

    with open(os.path.join(os.path.dirname(args.ckpt),'val_perf.json'),'w') as f:
        json.dump(val_perf,f)
    with open(os.path.join(os.path.dirname(args.ckpt),'test_perf.json'),'w') as f:
        json.dump(test_perf,f)


def evaluate(model,dataset,sp,device):
    model.eval()

    translates = []
    ground_truths = []
    copies = []
    labels = []
    
    with torch.no_grad():
        for src,tgt,label in dataset:
            src,tgt,label = src.to(device), tgt.to(device),label.to(device)

            src = src.unsqueeze(0)
            src_mask = model.make_src_mask(src)

            enc_src = model.encoder(src,src_mask)

            tgt_indices = [dataset.bos_id]

            # Generate sequence iteratively
            for i in range(500):
                tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)

                tgt_mask = model.make_tgt_mask(tgt_tensor)
                output,attention = model.decoder(tgt_tensor,enc_src,tgt_mask,src_mask)
                
                pred_token = output.argmax(2)[:,-1].item()
                tgt_indices.append(pred_token)

                if pred_token == dataset.eos_id:
                    break
            
            translate = sp.id_to_piece(tgt_indices[1:])
            # Without location info
            copy = sp.id_to_piece(src[0].tolist()[1:-1])
            # With location info : second token is location token
            # copy = sp.id_to_piece(src[0].tolist()[2:-1])
            ground_truth = sp.id_to_piece(tgt.tolist()[1:-1])

            translates.append(translate)
            copies.append(copy)
            ground_truths.append([ground_truth])
            labels.append(label.item())

            if len(labels)%1000==0:
                print(f"[{len(labels)}/{len(dataset)}] Sequence Processed.")
        
    perf_dict = get_performance(translates,copies,ground_truths,labels)

    return perf_dict

def get_performance(translates,copies,ground_truths,labels):
    perf_dict = {}
    # Total Bleu Score
    baseline = corpus_bleu(ground_truths,copies)
    model = corpus_bleu(ground_truths,translates)
    perf_dict['Baseline Total'] = baseline
    perf_dict['Model Total'] = model

    # Bleu Score on each location
    labels = np.array(labels)
    translates = np.array(translates,dtype=object)
    copies = np.array(copies,dtype=object)
    ground_truths = np.array(ground_truths,dtype=object)

    for location in np.unique(labels):
        indices = np.where(labels==location)[0]
        
        local_translate = translates[indices].tolist()
        local_copy = copies[indices].tolist()
        local_gt = ground_truths[indices].tolist()

        baseline = corpus_bleu(local_gt,local_copy)
        model = corpus_bleu(local_gt,local_translate)
        perf_dict[f'Baseline {location}'] = baseline
        perf_dict[f'Model {location}'] = model

    return perf_dict


if __name__ == '__main__':
    main()
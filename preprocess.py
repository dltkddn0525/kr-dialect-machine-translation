# ref : https://github.com/kakaobrain/jejueo/blob/master/translation/bpe_segment.py
import argparse
import pandas as pd
import os
import sentencepiece as spm
import codecs
import pickle


parser = argparse.ArgumentParser(description='Transformer dialect machine translation')
parser.add_argument('--data-dir', default='/nas/datahub/kr-dialect/dataset',type=str,
                    help='path to data of specific domain')
parser.add_argument('--vocab-size', default=4000,type=int,
                    help='Vocab Size')
args = parser.parse_args()

def train_bpe(fpath, vocab_size=4000):
    dir = os.path.dirname(fpath)
    train = f'--input={fpath} \
              --normalization_rule_name=identity \
              --model_prefix={dir}/bpe_{vocab_size} \
              --character_coverage=1.0 \
              --vocab_size={vocab_size} \
              --model_type=bpe \
              --unk_id=0 \
              --bos_id=1 \
              --eos_id=2 \
              --pad_id=3 \
              --unk_piece=<unk> \
              --bos_piece=<bos> \
              --eos_piece=<eos> \
              --pad_piece=<pad>'
    spm.SentencePieceTrainer.Train(train)

def main():
    data_dir = args.data_dir

    # Get dialect dataset(train)
    train_df = pd.read_csv(f"{data_dir}/train.csv",encoding='utf-8')
    train_dialect = train_df['dialect_form'].values.tolist()
    train_standard = train_df['standard_form'].values.tolist()

    with codecs.open(f"{data_dir}/bpe.train",'w','utf8') as fout:
        fout.write("\n".join(train_dialect + train_standard))

    # Train BPE
    train_bpe(f'{data_dir}/bpe.train',vocab_size=args.vocab_size)

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(f'{data_dir}/bpe_{args.vocab_size}.model')

    # Save Vocab
    vocab = {}
    for id_ in range(sp.get_piece_size()):
        token = sp.id_to_piece(id_)
        vocab[id_] = token

    with open(f'{data_dir}/vocab_dict_{args.vocab_size}.pickle','wb') as f:
        pickle.dump(vocab,f)


if __name__ == '__main__':
    main()

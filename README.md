# kr-dialect-machine-translation

1. Make data


```
python make_data.py --data <path to data>
```

- data dir
```
/nas/datahub/kr-dialect/train.csv
                        val.csv
                        test.csv
```

2. Train tokenizer(sentencepiece)

```
python preprocess.py --data-dir <path to data>
```
Result is as following.
```
data-dir/ bpe.model          # Tokenizer trained on train.csv
          bpe.train          
          bpe.vocab           
          vocab_dict.pickle  # Dictionary {idx : token}
```

3. Train Transformer Model
```
CUDA_VISIBLE_DEVICES=0 python main.py \
--data-dir <path to data> \
--save-path <path to save model & log>
```
- `data-dir` : `bpe.model` must be in this path.

### TODO
- validation code in main.py
- Test code(inference)

### Reference
- <href>https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

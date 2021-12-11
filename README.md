# kr-dialect-machine-translation

이 Repository는 Transformer를 활용하여 한국의 지역 방언을 표준어로 번역하는 기계 번역 모델 학습 및 테스트 코드를 포함하고 있습니다. 단계별 절차는 아래를 참고해주시기 바랍니다.

<hr>

## 1. Prepare Dataset

- 본 프로젝트에서는 AI hub의 지역별 방언 발화 데이터(경상도,전라도,충청도,제주도)를 사용했습니다.

<href>https://aihub.or.kr/aihub-data/natural-language/about

다음 코드를 통해 train, validation, test로 나누어줍니다.

```
python dataset/make_data.py --data <path to downloaded data>
```
  
- 데이터 셋을 다운받은 경로를 다음 코드를 통해 train,validation,test csv파일로 변환합니다.
- 이후, 지역별 train,validation,test를 각각 하나의 통합된 csv파일로 변환해주시기 바랍니다.
- 최종적인 데이터 형태는 다음과 같이 준비되어야 합니다.

```
data dir / train.csv
         / val.csv
         / test.csv
```

## 2. Train tokenizer(sentencepiece)

- Google에서 제공하는 sentencepiece tokenizer를 train data(표준어 + 방언)에 대해 학습하고 저장합니다. 이후 Transformer 모델 학습 시, 저장된 tokenizer를 활용하게 됩니다.

```
python preprocess.py --data-dir <path to data> --vocab-size 4000
```
Result is as following.
```
data-dir/ bpe_{vocab size}.model          # Tokenizer trained on train.csv
          bpe.train          
          bpe_{vocab size}.vocab           
          vocab_dict_{vocab size}.pickle  # Dictionary {idx : token}
```

## 3. Train Transformer Model

- Transformer 모델을 학습합니다. Configuration은 를 참고해주시기 바랍니다.

```
CUDA_VISIBLE_DEVICES=0 python main.py \
--data-dir <path to data> \
--save-path <path to save model & log> \
--use-loc False
```

- `data-dir` : `bpe_{vocab-size}.model` must be in this path.
- `save-path` : 학습된 모델(`last_model.pth`)이 이 경로에 저장됩니다.
- `use-loc`  : Location label을 활용할지 안할지를 결정하는 argument입니다.

## 4. Evaluate Model

- 학습된 Transformer 모델을 평가합니다. Validation ,Test set에 대한 BLEU score를 계산합니다.

```
CUDA_VISIBLE_DEVICES=0 python test.py \
--data-dir <path to data> \
--ckpt <path to model checkpoint>
```

- 전체 및 각 지역에 대한 BLEU score가 계산된 후 `json` 파일로 `ckpt` 에 입력한 경로에 저장됩니다.

<hr>

## Arguments
- `main.py`에 속하는 argument는 다음과 같습니다.

|Args|Description|Default|Type|
|----|-----------|-------|----|
|data-dir|Path to directory of dataset|'./data'|str|
|save-path|Path to save model|'./result'|str|
|batch-size|Training batch size|128|int|
|epoch|Training epoch|30|int|
|vocab-size|Vocab size|4000|int|
|hidden-dim|Token Embedding Dimension|256|int|
|enc-layer|Number of Transformer encoder block|6|int|
|dec-layer|Number of Transformer decoder block|6|int|
|enc-head|Number of attention head in encoder layer|8|int|
|dec-head|Number of attention head in decoder layer|8|int|
|use-loc|Whether to use location information|False|str2bool|

## Performance
- Baseline : Source 문장을 똑같이 번역했을 때 측정되는 BLEU score(번역 성능의 lower bound)
- Transformer : 전통적인 Transformer 모델의 BLEU score
- Transformer+location : Location information을 활용한 Transformer 모델의 BLEU score

|Method|Validation|Test|
|------|----------|----|
|Baseline|65.35|65.11|
|Transformer|81.12|80.58|
|Transformer+location|81.72|81.16|

<hr>

### Reference
- <href>https://github.com/ndb796/Deep-Learning-Paper-Review-and-Practice/blob/master/code_practices/Attention_is_All_You_Need_Tutorial_(German_English).ipynb

#%%
import warnings

import torch
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification,
                          AutoModelWithLMHead, AutoTokenizer, pipeline)

warnings.filterwarnings(action='ignore')
from transformers import logging

logging.set_verbosity_error()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
from googletrans import Translator

# %% load models(MCQ)
translator = Translator()

summarize_tokenizer = AutoTokenizer.from_pretrained("t5-small")
paraphrase_tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")

summarize_model = AutoModelWithLMHead.from_pretrained("t5-small")
paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
# add the EOS token as PAD token to avoid warnings
gpt2_model = AutoModelWithLMHead.from_pretrained("gpt2", pad_token_id=gpt2_tokenizer.eos_token_id)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')

summarize_model.to(DEVICE)
paraphrase_model.to(DEVICE)
gpt2_model.to(DEVICE)
bert_model.to(DEVICE)

# %% load models (WH)
## 정답 추출
kw_model = KeyBERT()

## 문제 생성
qg_tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator", use_fast=False)
qg_model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
qg_model.to(DEVICE)

## 문제 평가 
qae_tokenizer = AutoTokenizer.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
qae_model = AutoModelForSequenceClassification.from_pretrained("iarfmoose/bert-base-cased-qa-evaluator")
qae_model.to(DEVICE)

## 오답 생성
unmasker = pipeline('fill-mask', model='bert-base-cased')

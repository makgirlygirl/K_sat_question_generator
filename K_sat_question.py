#%%
## import package
import random

import torch
from keybert import KeyBERT

from K_sat_function import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import re

#%%
from model import bert_model, gpt2_model, gpt2_tokenizer

#%%
question_dict={'passageID':None,
                'question_type':None,
                'question':None, 
                'answer':None,
                'd1':None, 'd2':None, 'd3':None, 'd4':None}
#%%
# 18, 20, 22(목적/요지/주제): 한국어 보기
# 23, 24, 41(주제/제목): 영어 보기
class Q1:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list):
        paraphrase=paraphrasing_by_transe(summary)
        sent_completion_dict=get_sentence_completions(paraphrase)
        return paraphrase, sent_completion_dict

    def distractors(self, sent_completion_dict:dict):
        distractors=[]
        distractor_cnt = 1
        for key_sentence in sent_completion_dict:
            if distractor_cnt == 6:
                break
            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences =[]
            false_sents = []
            for partial_sent in partial_sentences:
                for repeat in range(10):
                    false_sents = generate_sentences(partial_sent, key_sentence)
                    if false_sents != []:
                        break
                false_sentences.extend(false_sents)
            distractors.extend(paraphrasing_by_transe(false_sentences[:1]))
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, q1_paraphrase, false_sentences, is_Korean=False):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]
        ## 랜덤으로 답 뽑기
        i=random.randint(1, 5)
        q1_paraphrase=q1_paraphrase[i-1:i]
        false_sentences2=false_sentences.copy()
        del false_sentences2[i]  # 4개
        ## 제목
        if '제목' in question[0]:
            q1_paraphrase=get_keyword_list(q1_paraphrase, max_word=5, top_n=1)
            false_sentences2=get_keyword_list(false_sentences2, max_word=5, top_n=1)
        ## 번역
        if is_Korean==True:
            q1_paraphrase=transe_kor(q1_paraphrase)
            false_sentences2=transe_kor(false_sentences2)
        question_dict['answer']=q1_paraphrase[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        return question_dict
#%%
# 26-28, 45(내용 일치/불일치): 영어 보기
class Q2:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list):
        paraphrase=paraphrasing_by_transe(summary)
        sent_completion_dict=get_sentence_completions(paraphrase)
        return paraphrase

    def distractors(self, paraphrase:list):
        distractors=[]
        distractor_cnt = 1
        wd=word_dict()
        for key_sentence in paraphrase:
            if distractor_cnt == 6:
                break

            keyword_list=get_keyword_list(key_sentence, max_word_cnt=2, top_n=1) 
            keyword=keyword_list[0]

            if len(keyword.split())==1:
                change_word=keyword
            else:
                change_word=random.choice(keyword.split())
            
            antonym_list=get_antonym_list(change_word, num_word=1)
            antonym_list=sum(antonym_list, [])
            distractor_sentence=key_sentence.replace(change_word, antonym_list[0])
            distractors.append(distractor_sentence)
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, q1_paraphrase, false_sentences):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]

        ## 적절하지 않은? 적절한?
        if '않은' not in question[0]:## 오답4개 정답1개가 되어야함
            q1_paraphrase2=false_sentences
            false_sentences2=q1_paraphrase

        ## 랜덤으로 답 뽑기
        i=random.randint(1, 5)
        q1_paraphrase=q1_paraphrase[i-1:i]
        del false_sentences2[i]  # 4개

        question_dict['answer']=q1_paraphrase[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        return question_dict
#%%


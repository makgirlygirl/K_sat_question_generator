#%%
## import package
import random

import torch
from keybert import KeyBERT

from K_sat_function import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def distractors(self, sent_completion_dict:dict, isTranse=True):
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
            q1_paraphrase=get_title(q1_paraphrase)
            false_sentences2=get_title(false_sentences2)
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
q1=Q1()
q1_question_type=1
q1_question=['다음 글의 제목으로 가장 적절한 것은?']
q1_summarize=q1.summarize(passage)
q1_paraphrase, q1_sent_completion_dict=q1.paraphrase(q1_summarize)
q1_distractors=q1.distractors(q1_sent_completion_dict)
# q1_dict_kor=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors,is_Korean=True)
q1_dict_eng=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors)
# %%
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
        return paraphrase, sent_completion_dict

    def distractors(self, sent_completion_dict:dict, isTranse=True):
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
        ## 번역
        if is_Korean==True:
            q1_paraphrase=transe_kor(q1_paraphrase)
            false_sentences2=transe_kor(false_sentences2)
        # print(false_sentences)
        question_dict['answer']=q1_paraphrase[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        return question_dict

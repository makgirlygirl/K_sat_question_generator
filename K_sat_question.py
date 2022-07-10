#%%
## import package
import random

import torch

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
# 18, 20, 22(목적/요지/주제/제목): 한국어 보기
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
            distractors.append(paraphrasing_by_transe(false_sentences[:1]))
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, answer, q1_paraphrase, is_Korean=False):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question
        ## 랜덤으로 답 뽑기
        i=random.randint(1, 5)
        ## 번역
        if is_Korean==True:
            answer=transe_kor(answer[i-1:i])
            false_sentences=transe_kor(false_sentences)
        print(answer)
        question_dict['answer']=answer
        question_dict['d1']=false_sentences[0]
        question_dict['d2']=false_sentences[1]
        question_dict['d3']=false_sentences[2]
        question_dict['d4']=false_sentences[3]

        return question_dict

#%%
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()

q1=Q1()
q1_summarize=q1.summarize(passage)
q1_paraphrase, q1_sent_completion_dict=q1.paraphrase(q1_summarize)
q1_distractors=q1.distractors(q1_sent_completion_dict)
#%%

#%%
q1_dict=q1.make_dict(passageID, 'q1', 'aaa',q1_paraphrase,q1_distractors)

#%%
print(q1_dict)

#%%
## 26-28, 45: 내용일치, 불일치
## 중심문장(sumy 5)찾고-distractor생성
#%%
## 32-34: 빈칸추론(문장)
## 중심문장 찾고(sumy 1)-distractor생성-지문 뚫기
#%%

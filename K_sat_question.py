#%%
## import package
import random

import torch
from keybert import KeyBERT
from tomlkit import key

from K_sat_function import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import re

#%%
from model import bert_model, gpt2_model, gpt2_tokenizer

#%%
question_dict_sample={'passageID':None,
                'question_type':None,
                'question':None, 
                'new_passage':None,
                'answer':None,
                'd1':None, 'd2':None, 'd3':None, 'd4':None}
#%% fin
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
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]
        
        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        q1_paraphrase2=[]
        false_sentences2=[]

        q1_paraphrase2.append(q1_paraphrase[num])

        for i in range(5):
            if i!=num:
                false_sentences2.append(false_sentences[i])

        ## 제목
        if '제목' in question[0]:
            q1_paraphrase2=get_keyword_list(q1_paraphrase2, max_word_cnt=5, top_n=1)
            false_sentences2=get_keyword_list(false_sentences2, max_word_cnt=5, top_n=1)
            is_Korean==False
        ## 번역
        if is_Korean==True:
            print('Kor')
            print('\n\n')
            q1_paraphrase2=transe_kor(q1_paraphrase2)
            false_sentences2=transe_kor(false_sentences2)

        print(q1_paraphrase2)
        print(false_sentences2)
        print(len(q1_paraphrase2))
        print(len(false_sentences2))
    
        question_dict['answer']=q1_paraphrase2[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        return question_dict
#%%
'''
q1=Q1()
q1_question_type=1
q1_list=['목적으로', '주장으로', '요지로', '제목으로']
q1_question=[f'다음 글의 {random.choice(q1_list)} 가장 적절한 것은?']
q1_summarize=q1.summarize(passage)
q1_paraphrase, q1_sent_completion_dict=q1.paraphrase(q1_summarize)
q1_distractors=q1.distractors(q1_sent_completion_dict)
q1_dict_kor=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors, is_Korean=True)
print(q1_dict_kor)
## dict가 계속 바뀌니까 한번 하고 디비에  넣고 한번 하고 디비에 넣고 해야함
q1_dict_eng=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors, is_Korean=False)
print(q1_dict_eng)
print(q1_dict_kor)'''
#%% fin
# 26-28, 45(내용 일치/불일치): 영어 보기/한글 보기
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

    ## 오답단어 4개 만들기
    def distractors(self, paraphrase:list):
        distractors=[]
        distractor_cnt = 1
        wd=word_dict()
        for key_sentence in paraphrase:
            if distractor_cnt == 6:
                break

            keyword_list=get_keyword_list(key_sentence, max_word_cnt=2, top_n=1) 
            keyword=keyword_list[0]
            # print(keyword)


            if len(keyword.split())==1:
                change_word=keyword
            else:
                change_word=random.choice(keyword.split())
            
            # print(change_word)
            antonym_list=get_antonym_list(change_word, num_word=1)
            # print(antonym_list)

            antonym_list=sum(antonym_list, [])
            if len(antonym_list)>0:
                distractor_sentence=key_sentence.replace(change_word, antonym_list[0])
            else: print('please check \'https://www.powerthesaurus.org/\'');return None

            distractors.append(distractor_sentence)
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, q1_paraphrase, false_sentences, is_Korean=False):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]

        ## 적절하지 않은? 적절한?
        if '않은' in question[0]:## 적절한거 고르기: 오답4 정답1
            q1_paraphrase=false_sentences
            false_sentences=q1_paraphrase

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)# a <= num <= b
        print(num)## 2
        # num=2
        q1_paraphrase2=[]
        false_sentences2=[]

        q1_paraphrase2.append(q1_paraphrase[num])

        for i in range(5):
            print(i)
            if i!=num:
                false_sentences[i]
                false_sentences2.append(false_sentences[i])
            if i==num:print('aaaa')

        print(false_sentences2)
        print(q1_paraphrase2)


        ## 번역
        if is_Korean==True:
            print('Kor')
            q1_paraphrase2=transe_kor(q1_paraphrase2)
            false_sentences2=transe_kor(false_sentences2)

        question_dict['answer']=q1_paraphrase2[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        return question_dict
#%%test q2 ->fin
'''
q2=Q2()
q2_question_type=2
q2_list=['적절한', '적절하지 않은']
q2_question=[f'윗글에 관한 내용으로 가장 {random.choice(q2_list)} 것은?']
q2_summarize=q2.summarize(passage)
q2_paraphrase=q2.paraphrase(q2_summarize)
q2_distractors=q2.distractors(q2_paraphrase)
q2_dict=q2.make_dict(passageID, q2_question_type, q2_question, q2_paraphrase, q2_distractors, is_Korean=False)

print(q2_dict)
'''


#%%
## 36-37, 43(순서(ABC)): 영어 보기
#%%test q3




#%%
## 32-34(빈칸추론(문장)): 영어 보기
class Q4:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    ## 중심문장 1문장 찾기
    def summarize(self, passage:str, num_sentence=1)->list:
        return summary(passage, num_sentences=num_sentence)
    
    
    def get_sentence_completions(self, summary:list):
        sent_completion_dict=get_sentence_completions(summary)
        return sent_completion_dict

    ## 오답문장 4개 만들기
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

    ## new_passage
    def make_new_passage(self, passage:str, summary:list)->str:
        for sentence in summary:
            if sentence in passage:
                space='_'*len(sentence)
                new_passage=passage.replace(sentence, space)
                return new_passage
            else: print('No result');return None

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, new_passage:str, q1_paraphrase, false_sentences, is_Korean=False):
        question_dict=question_dict_sample.copy()
        
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]
        
        ## 새로운 passage
        question_dict[new_passage]=new_passage

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        q1_paraphrase2=[]
        false_sentences2=[]

        q1_paraphrase2.append(q1_paraphrase[num])

        for i in range(5):
            if i!=num:
                false_sentences2.append(false_sentences[i])

                
    
        question_dict['answer']=q1_paraphrase2[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        
        return question_dict

#%% test q4
q4=Q4()
q4_question_type=4
q2_list=['적절한', '적절하지 않은']
q2_question=[f'윗글에 관한 내용으로 가장 {random.choice(q2_list)} 것은?']
q2_summarize=q2.summarize(passage)
q2_paraphrase=q2.paraphrase(q2_summarize)
q2_distractors=q2.distractors(q2_paraphrase)
q2_dict=q2.make_dict(passageID, q2_question_type, q2_question, q2_paraphrase, q2_distractors)

print(q2_dict)




#%%
## 빈칸추론 (단어)?: 3140 등등->중심문장->단어
## 단어두개뽑는거솓?

class Q5:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    ## 중심문장 1문장 찾기
    def summarize(self, passage:str, num_sentence=1)->list:
        return summary(passage, num_sentences=num_sentence)

    ## 중심문장의 keyword 찾기
    ## top_n=1 : 그냥 빈칸
    ## top_n=2: 40번->나중에생각하자
    def get_ans(self, summarize:list, top_n=1)->list:
        return get_keyword_list(summarize, max_word_cnt=1, top_n=top_n)

    ## 오답단어 4개 만들기
    def distractors(self, ans:list)->list:
        wd=word_dict()
        if len(ans)==1:
            antonym_list=get_antonym_list(ans, num_word=2)
            antonym_list=sum(antonym_list, [])
            for antonym in antonym_list:
                antonym_list2=get_synonym_list(antonym,num_word=1)
                antonym_list.append(antonym_list2[0])
        else:# 40번->나중에생각하자
            None

        return antonym_list

    ## new_passage
    def make_new_passage(self, passage:str, summary:list, ans: list)->str:
        if len(ans)==1:
            ans=ans[0]
            for sentence in summary:
                if sentence in passage:
                    if sentence==summary[0]:
                        space='_'*len(ans)
                        new_passage=passage.replace(ans, space)
                else: print('No result');new_passage=None
        else: # 40번->나중에생각하자
            None
        return new_passage


    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, new_passage:str, q1_paraphrase, false_sentences, is_Korean=False):
        question_dict=question_dict_sample.copy()
        
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]
        
        ## 새로운 passage
        question_dict[new_passage]=new_passage

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        q1_paraphrase2=[]
        false_sentences2=[]

        q1_paraphrase2.append(q1_paraphrase[num])

        for i in range(5):
            if i!=num:
                false_sentences2.append(false_sentences[i])
    
        question_dict['answer']=q1_paraphrase2[0]
        question_dict['d1']=false_sentences2[0]
        question_dict['d2']=false_sentences2[1]
        question_dict['d3']=false_sentences2[2]
        question_dict['d4']=false_sentences2[3]

        
        return question_dict

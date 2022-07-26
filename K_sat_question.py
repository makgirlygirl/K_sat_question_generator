#%%
## import package
import random
from hashlib import new

import torch
from keybert import KeyBERT
from tomlkit import key
from transformers import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP

from K_sat_function import *

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import re

#%%
from model import bert_model, gpt2_model, gpt2_tokenizer

#%%
###### 나중에 지울거

f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()
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

    ## 오답 단어 4개 만들기->오답 문장 만들기
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

#%% fin
## 36-37, 43(순서(ABC)): 영어 보기
class Q3:
    def __init__(self):
        None
    
    #문장단위로 쪼개고 ABC를 랜덤하게 234번째 문단 앞에 붙히기
    def separate(self,passage:str):
        # temp1=passage        #passage는 그대로 냅둘라고
        temp=passage.split('.') # 마침표 기준. 리스트로 쪼갬
        l=len(temp)
        # print(temp)
        num=l//4
        new_passage_lst=[]
        new_passage_lst.append(temp[:num])   #0문단-> 얘는 처음에 주어짐. 처음~1/4까지
        new_passage_lst.append(temp[num:2*num])     #1문단-> 1/4 다음 문장에서 1/2문장까지
        new_passage_lst.append(temp[2*num:3*num])   #~
        new_passage_lst.append(temp[3*num:])   #~


        # new_passage_lst[0]=sum(new_passage_lst[0][0],new_passage_lst[0][1])
        # print(new_passage_lst)
        for i in range(4):
            new_passage_lst[i]=' '.join(new_passage_lst[i])

        #정답을 랜덤하게 배치
        answer_list=[['A','B','C'],['A','C','B'],['B','A','C'],['B','C','A'],['C','A','B'],['C','B','A']]
        select=random.sample(answer_list,5)  #위에 6개 중 5개 랜덤하게 선택됨
        ans=select[0]   #첫번째 거가 정답임
        ans_num=[]
        for j in range(0,3):
            if ans[j]=='A': ans_num.append(1)
            if ans[j]=='B': ans_num.append(2)
            if ans[j]=='C': ans_num.append(3)

        distractors=select[1:5]   #나머지가 distractor 됨

        show=[]                 #정답 순서 따라 new_passage_lst를 show에 재배치
        for j in range(len(new_passage_lst)):
            show.append(new_passage_lst[j])

        show[ans_num[0]]=new_passage_lst[1]
        show[ans_num[1]]=new_passage_lst[2]
        show[ans_num[2]]=new_passage_lst[3]

        new_passage=''
        new_passage=str(show[0])+'\n'+'(A)'+'\t'+str(show[1])+'\n'+'(B)'+'\t'+str(show[2])+'\n'+'(C)'+'\t'+str(show[3])
             

        return new_passage, ans, distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, passage:str)->dict:
        question_dict=question_dict_sample
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question[0]
        new_passage, ans, distractor=self.separate(passage)

        question_dict['new_passage'] = new_passage
        question_dict['answer']=ans
        question_dict['d1']=distractor[0]   ##이게 출력될 때 A B C 이렇게 나올텐데 A->B->C로 애초에 저장할까?
        question_dict['d2']=distractor[1]
        question_dict['d3']=distractor[2]
        question_dict['d4']=distractor[3]

        return question_dict
#%%test q3->fin
'''
q3=Q3()
q3_question_type=3
q3_question=[f'문단을 올바른 순서로 배치']
passageID=2

#4문단으로 분리, 1문단은 주어짐. 234 랜덤으로 ABC 부여
#답안 생성, A B C 랜덤하게 배치, 서로 겹치지 않게


q3_dict=q3.make_dict(passageID, q3_question_type, q3_question, passage)
print(q3_dict)
print(q3_dict['question'])
print(q3_dict['new_passage'])
print(q3_dict['answer'])'''
#%%
# def _get_soup_object(url, parser="html.parser"):
#     headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36',
#     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#     }
#     return BeautifulSoup(requests.get(url,headers=headers).text, parser)

# class word_dict(object):
#     def __init__(self, *args):
#         try:
#             if isinstance(args[0], list):
#                 self.args = args[0]
#             else:
#                 self.args = args
#         except:
#             self.args = args
    
#     def getSynonyms(self, num_word, formatted=False):
#         return [self.synonym(term, num_word, formatted) for term in self.args]
        
#     def getAntonyms(self,num_word, formatted=False):
#             return [self.antonym(term, num_word, formatted) for term in self.args]

#     def synonym(self, term, num_word, formatted=False):
#         if len(term.split()) > 1:
#             print("Error: A Term must be only a single word")
#         else:
#             try:
#                 data = _get_soup_object("https://www.powerthesaurus.org/"+term+"/synonyms")
#                 # section = data.findAll('a', {'class': "ch_at ch_ci aaa_at"})[:num_word]
#                 section = data.findAll('a', {'class': "cl_az cl_cm z4_az"})[:num_word]
#                 synonyms=[s.text.strip() for s in section]
#                 # print(synonyms)
#                 if formatted:
#                     return {term: synonyms}
#                 return synonyms
#             except: None

#     def antonym(self, term, num_word, formatted=False):
#         if len(term.split()) > 1:
#             print(term)
#             print("Error: A Term must be only a single word")
#         else:
#             try:
#                 data = _get_soup_object("https://www.powerthesaurus.org/"+term+"/antonyms")
#                 # section = data.findAll('a', {'class': "ch_at ch_ci aaa_at"})[:num_word]
#                 section = data.findAll('a', {'class': "cl_az cl_cm z4_az"})[:num_word]
#                 antonyms=[s.text.strip() for s in section]
#                 if formatted:
#                     return {term: antonyms}
#                 return antonyms
#             except:None

#%%
## 31(빈칸추론(단어)): 영어 보기
class Q4:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    ## 중심문장 1문장 찾기(지문내의 문장)
    def summarize(self, passage:str, num_sentence=1)->list: # 1문장
        return summary(passage, num_sentences=num_sentence)
    
    def paraphrase(self, summary:list)->list:   # 1문장
        paraphrase=paraphrasing_by_transe(summary)
        return paraphrase

    def get_answer(self, paraphrase:list)->str: # 1문장 들어가서 단어 1개 나옴
        answer_list=get_keyword_list(paraphrase, max_word_cnt=1, top_n=1)
        return answer_list[0]

    ## 오답 단어 4개 만들기
    ############# 수정ㅇ하기!!
    def distractors(self, answer:str)->list:
        wd=word_dict()
        a=get_antonym_list(answer, num_word=1)
        print(a)
        # print('\n\n')
        b=wd.synonym(answer, num_word=3)
        # print(b)
        distractors=sum(get_antonym_list(answer, num_word=1), get_synonym_list(answer, num_word=3))
        return distractors

    ## new_passage
    def make_new_passage(self, passage:str, summarize:str, paraphrase:str, answer:str): #fin, str/None 리턴
        if summarize in passage:
            space='_'*int(len(answer)*0.6)
            new_passage=passage.replace(summarize, paraphrase).replace(answer, space)
            return new_passage  ## str
        else: print('No result');return None


    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID, question_type, question, new_passage:str, answer, distractors):
        question_dict=question_dict_sample.copy()
        
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type    # 4
        question_dict['question'] = question[0] # '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오.'
        ## 새로운 passage
        question_dict[new_passage]=new_passage
    
        question_dict['answer']=answer
        question_dict['d1']=distractors[0]
        question_dict['d2']=distractors[1]
        question_dict['d3']=distractors[2]
        question_dict['d4']=distractors[3]

        
        return question_dict

#%%
q4=Q4()
q4_question_type=4
q4_question=['다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오']
q4_summarize=q4.summarize(passage)
q4_paraphrase=q4.paraphrase(q4_summarize)
q4_answer=q4.get_answer(q4_paraphrase)
# #%%
# print(q4_summarize)
# print(q4_paraphrase)
# print(q4_answer)
#%%
q4_new_passage=q4.make_new_passage(passage, q4_summarize[0], q4_paraphrase[0], q4_answer)
#%%
q4_distractors=q4.distractors(q4_answer)

print(q4_distractors)
#%%
q4_dict=q4.make_dict(passageID, q4_question_type, q4_question, q4_new_passage, q4_answer, q4_distractors)

print(q4_dict)
#%%
## 30, 42
## 다음글의밑줄친부분중,문맥상낱말의쓰임이적절하지 않은 것은?

class Q5:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def get_answer(self, paraphrase:list)->str: # 5문장 들어가서 단어 5개 나옴
        answer_list=get_keyword_list(paraphrase, max_word_cnt=1, top_n=1)
        return answer_list[0]
    
    ## new_passage
    def make_new_passage(self, passage:str, summarize:list, answer:list)->str: 
        # new_sum=[]
        for i in range(len(summarize)):
            if answer[i] in summarize[i]:
                space='_'*int(len(answer[i])*0.6)
                new_sum=summarize[i].replace(answer[i], space)
            if summarize[i] in passage:
                new_passage=passage.replace(summarize[i], new_sum)
        return new_passage
                


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


#%%
'''## 32-34(빈칸추론(문장)): 영어 보기
## 핵심문장 3개-> 중심단어->그게 포함된 구/절
class Q5:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    ## 중심문장 3문장 찾기(지문내의 문장)
    def summarize(self, passage:str, num_sentence=3)->list:
        return summary(passage, num_sentences=num_sentence)
    
    ## 문장을 구/절 로 나눔
    def sentence_split(self, summarize:list):
        None    

    ## 문장에서 키워드 추출 -> 키워드가 포함된 
    def get_answer(self, summarize:list)->str:
        for sentence in summarize:
            keyword=get_keyword_list(sentence, max_word_cnt=1, top_n=1)
        return answer

    ## 오답문장 4개 만들기
    # def distractors(self, sent_completion_dict:dict, answer:str):## 
    def distractors(self, sent_completion_dict:dict):
        distractors=[]
        distractor_cnt = 1
        for key_sentence in sent_completion_dict:
            print(distractor_cnt)
            if distractor_cnt == 6:
                break
            partial_sentences = sent_completion_dict[key_sentence]
            false_sentences_list =[]
            false_sentence = []
            for partial_sent in partial_sentences:
                for repeat in range(10):
                    false_sentence = generate_sentences_for_blank_inference(partial_sent, key_sentence)
                    if false_sentence != []:
                        break
                false_sentences_list.extend(false_sentence)
                print(false_sentences_list)
                print('\n\n')
            distractors.extend(paraphrasing_by_transe(false_sentences_list[:1]))
            distractor_cnt += 1
        return distractors

    ## new_passage
    def make_new_passage(self, passage:str, answer:list): #fin, str/None 리턴
        for sentence in answer:
            if sentence in passage:
                space='_'*int(len(sentence)*0.6)
                new_passage=passage.replace(sentence, space)
                return new_passage
            else: print('No result');return None

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID, question_type, question, new_passage:str, q1_paraphrase, false_sentences):
        question_dict=question_dict_sample.copy()
        
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type    # 4
        question_dict['question'] = question[0] # '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오.'
        
        ## 새로운 passage
        question_dict[new_passage]=new_passage

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        # print(num)
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
'''

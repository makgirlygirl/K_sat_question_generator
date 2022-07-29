#%%
## import package
import random
import re
from hashlib import new

import torch
from keybert import KeyBERT
from tomlkit import key
from transformers import SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP

from K_sat_function import *
from model import bert_model, gpt2_model, gpt2_tokenizer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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
#%% 18, 20, 22(목적/요지/주제): 한국어 보기, 23, 24, 41(주제/제목): 영어 보기
class Q1:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
        self.question_type=1
        self.qlist=['목적으로', '주장으로', '요지로', '제목으로']
        self.question=f'다음 글의 {random.choice(self.qlist)} 가장 적절한 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list):
        paraphrase=paraphrasing_by_transe(summary)  ## list
        sent_completion_dict=get_sentence_completions(paraphrase)   ## dict
        return paraphrase, sent_completion_dict

    def get_false_paraphrase(self, sent_completion_dict:dict)->list:
        false_paraphrase=[]
        false_paraphrase_cnt = 1
        for key_sentence in sent_completion_dict:
            if false_paraphrase_cnt == 6:
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
            false_paraphrase.extend(paraphrasing_by_transe(false_sentences[:1]))
            false_paraphrase_cnt += 1
        return false_paraphrase

    def make_dict(self, passageID, is_Korean=False)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################
        
        summarize=self.summarize(passage)   ## list
        paraphrase, completion_dict=self.paraphrase(summarize)  ## list, dict
        false_paraphrase=self.get_false_paraphrase(completion_dict) ## list

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)
        answer=paraphrase[num]  ## str
        dist_list=false_paraphrase[:num]+false_paraphrase[num+1:]   ## list

        ## 제목
        if '제목' in self.question:
            answer=get_keyword_list(answer, max_word_cnt=5, top_n=1)    ## input: str -> output: str
            dist_list=get_keyword_list(dist_list, max_word_cnt=5, top_n=1)  ## input: list -> output: list
            is_Korean==False

        ## 번역
        if is_Korean==True:
            answer=transe_kor(answer)   ## input: str -> output: str
            dist_list=transe_kor(dist_list) ## input: list -> output: list
    
        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return question_dict

#%% 26-28, 45(내용 일치/불일치): 영어 보기/한글 보기
class Q2:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
        self.question_type=2
        self.qlist=['적절한', '적절하지 않은']
        self.question=f'윗글에 관한 내용으로 가장 {random.choice(self.qlist)} 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list)->list:
        paraphrase=paraphrasing_by_transe(summary)
        sent_completion_dict=get_sentence_completions(paraphrase)
        return paraphrase

    def get_false_sentence(self, paraphrase:list)->list:
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

    def make_dict(self, passageID, is_Korean=False)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        summarize=self.summarize(passage)   ## list
        key_sentences=self.paraphrase(summarize)    ## list
        false_sentences=self.get_false_sentence(key_sentences)  ## list

        ## 적절하지 않은? 적절한?
        if '않은' in self.question:   ## 적절하지 않은 것 고르기: 적절 4 안적절 1
            ans_list=false_sentences.copy()
            dist_list=key_sentences.copy()
        else:   ## 적절한 것 고르기: 적절 1 안적절 4
            ans_list=key_sentences.copy()
            dist_list=false_sentences.copy()

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)# a <= num <= b
        answer=ans_list[num]    ## 정답
        del dist_list[num]  ## 오답 list 수정

        ## 번역
        if is_Korean==True:
            print('Kor')
            answer=transe_kor(answer)   ## input: str -> output: str
            dist_list=transe_kor(dist_list) ## input: list -> output: list

        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return question_dict

#%% 36-37, 43(순서(ABC)): 영어 보기
class Q3:
    def __init__(self):
        self.question_type=3
        self.question='주어진 글 다음에 이어질 글의 순서로 가장 적절한 것을 고르시오.'

    def separate(self,passage:str):
        ## 문장단위로 쪼개고 ABC를 랜덤하게 234번째 문단 앞에 붙히기
        ## 4문단으로 분리, 1문단은 주어짐. 234 랜덤으로 ABC 부여
        ## 답안 생성, A B C 랜덤하게 배치, 서로 겹치지 않게
        # temp1=passage        #passage는 그대로 냅둘라고
        temp=passage.split('.') # 마침표 기준. 리스트로 쪼갬
        l=len(temp)
        # print(temp)
        num=l//4
        new_passage_lst=[]
        new_passage_lst.append(temp[:num])   ## 0문단-> 얘는 처음에 주어짐. 처음~1/4까지
        new_passage_lst.append(temp[num:2*num])     ## 1문단-> 1/4 다음 문장에서 1/2문장까지
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
             

        return new_passage, ans, distractors    ## str, str, list

    def make_dict(self, passageID)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        new_passage, ans, distractor=self.separate(passage)

        question_dict['new_passage'] = new_passage
        question_dict['answer']=ans
        question_dict['d1']=distractor[0]   ##이게 출력될 때 A B C 이렇게 나올텐데 A->B->C로 애초에 저장할까?
        question_dict['d2']=distractor[1]
        question_dict['d3']=distractor[2]
        question_dict['d4']=distractor[3]

        return question_dict
#%% 31(빈칸추론(단어)): 영어 보기
class Q4:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
        self.question_type=4
        self.question='다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오'

    def summarize(self, passage:str, num_sentence=1)->list: ## 중심문장 1문장(지문 내부 문장)
        return summary(passage, num_sentences=num_sentence)
    
    def paraphrase(self, summary:list)->list:   ## 1문장
        paraphrase=paraphrasing_by_transe(summary)
        return paraphrase

    def get_answer(self, paraphrase:list)->str: ## 1문장 들어가서 키워드 1개 나옴
        answer_list=get_keyword_list(paraphrase, max_word_cnt=1, top_n=1)
        return answer_list[0]

    def get_distractors(self, answer:str)->list:    ## 오답 단어 4개 만들기
        wd=word_dict()
        antonym_list=get_antonym_list(answer, num_word=1)  ## 쉬운 오답(반의어): 1개
        synonym_list=get_synonym_list(answer, num_word=3)  ## 어려운 오답(유의어): 3개
        distractors=sum(antonym_list,[]) ## sum( 덧셈할 것, 처음에 더할 것)
        distractors=sum(synonym_list,distractors) ## sum( 덧셈할 것, 처음에 더할 것)

        return distractors

    def make_new_passage(self, passage:str, summarize:str, paraphrase:str, answer:str)->str: #fin, str/None 리턴
        if summarize in passage:
            space='_'*int(len(answer)*0.6)
            new_passage=passage.replace(summarize, paraphrase).replace(answer, space)
            return new_passage  ## str
        else: print('No result');return None

    def make_dict(self, passageID):
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question   ## '다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오'

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        summarize=self.summarize(passage)  ## list
        paraphrase=self.paraphrase(summarize)   ## list
        answer=self.get_answer(paraphrase)  ## str
        dist_list=self.get_distractors(answer)    ## list
        ##  summarize[0]:str, paraphrase[0]:str, answer:str
        new_passage=self.make_new_passage(passage, summarize[0], paraphrase[0], answer)   ##str


        ## 새로운 passage
        question_dict['new_passage']=new_passage
    
        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        
        return question_dict

#%% 30, 42(적절하지 않은 단어)
class Q5:
    def __init__(self):
        self.gpt2_tokenizer = gpt2_tokenizer
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
        self.question_type=5
        self.q_list=['한', '하지 않은']
        self.question=f'다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절{random.choice(self.q_list)} 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def get_keyword(self, paraphrase:list)->list: ## 5문장 들어가서 단어 5개 나옴
        keyword_list=get_keyword_list(paraphrase, max_word_cnt=1, top_n=1)
        return keyword_list

    def get_keyword_antonym(self, keyword_list:list)->list: ## 단어 5개 들어가서 단어 5개 나옴
        antonym_list=[]
        wd=word_dict()
        for i in keyword_list:
            antonym_list.append(wd.antonym(i,1)[0])  ## 반의어
        return antonym_list

    def make_new_passage(self, passage:str, summarize_list:list, old_list:list, new_list:list)->str:
        new_passage=''+passage
        for i in range(len(summarize_list)):
            summ=summarize_list[i]
            old=old_list[i]
            new=new_list[i]
            # print((summ))
            # print((old))
            # print((new))

            if old.lower() in summ.lower(): ## 문장이 바뀌는 경우
                new_summ=summ.replace(old, new)
                if summ in new_passage:
                    new_passage=new_passage.replace(summ, new_summ)
        return new_passage

    def make_dict(self, passageID)->dict:
        question_dict=question_dict_sample.copy()
        flag=True
        question_dict['passageID']=int(passageID)

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        question_dict['question_type']=self.question_type
        question_dict['question'] =self.question

        summarize=self.summarize(passage)  ## list
        keyword=self.get_keyword(summarize) ## list
        keyword_antonym=self.get_keyword_antonym(keyword)   ## list


        ## 적절하지 않은? 적절한?
        if '않은' in self.question:   ## 적절하지 않은 것 고르기: 적절 4 안적절 1 -> 하나만 바꿔주면됨
            ans_list=keyword_antonym.copy()
            dist_list=keyword.copy()
            flag=False
        else:   ## 적절한 것 고르기: 적절 1 안적절 4 -> 오답4개 바꿔주어야함
            ans_list=keyword.copy()
            dist_list=keyword_antonym.copy()

        ## 랜덤으로 답 뽑기
        num=random.randint(0, 4)    ## a <= num <= b
        answer=ans_list[num]    ## 정답
        del dist_list[num]  ## 오답 list 수정
        
        if flag==False: ## 1개만 바꿔주면 됨
            summarize_list=[summarize[num]]
            old_list=[keyword[num]]
            new_list=[answer]
        else:   ## 4개를 바꿔주어야 함
            summarize_list=summarize[:num]+summarize[num+1:]
            old_list=keyword[:num]+keyword[num+1:]
            new_list=keyword_antonym[:num]+keyword_antonym[num+1:]

            # print(len(summarize_list))
            # print(len(old_list))
            # print(len(new_list))

        ## 새로운 passage 만들기
        question_dict['new_passage']=self.make_new_passage(passage, summarize_list, old_list, new_list)
    
        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return question_dict

#%%


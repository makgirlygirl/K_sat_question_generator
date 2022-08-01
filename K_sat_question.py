#%%
## import package
import random
import re
from hashlib import new

import nltk
import torch
from keybert import KeyBERT
from tomlkit import key
from transformers import pipeline

from K_sat_function import *

# from model import (bert_model, gpt2_model, gpt2_tokenizer, paraphrase_model,
#                    paraphrase_tokenizer, summarize_model, summarize_tokenizer)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#%%
####################################
#  나중에 지울거
####################################

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
        self.question_type=2
        self.qlist=['적절한', '적절하지 않은']
        self.question=f'윗글에 관한 내용으로 가장 {random.choice(self.qlist)} 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list)->list:
        paraphrase=paraphrasing_by_transe(summary)
        # sent_completion_dict=get_sentence_completions(paraphrase)
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
            # print('Kor')
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
        self.question_type=4
        self.question='다음 빈칸에 들어갈 말로 가장 적절한 것을 고르시오'

    def summarize(self, passage:str, num_sentence=1)->list: ## 중심문장 1문장(지문 내부 문장)
        return summary(passage, num_sentences=num_sentence)
    
    def paraphrase(self, summary:list)->list:   ## 1문장
        return paraphrasing_by_transe(summary)

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
        self.question_type=5
        self.question='다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def get_keyword(self, paraphrase:list)->list: ## 5문장 들어가서 단어 5개 나옴
        return get_keyword_list(paraphrase, max_word_cnt=1, top_n=1)

    def get_keyword_antonym(self, keyword_list:list)->list: ## 단어 5개 들어가서 단어 5개 나옴
        antonym_list=[]
        wd=word_dict()
        for i in keyword_list:
            antonym_list.append(wd.antonym(i,1)[0])  ## 반의어
        return antonym_list

    def make_new_passage(self, passage:str, summarize_list:list, origin_list:list, new_list:list ,answer:int)->str:
        new_passage=''+passage
        for i in range(len(summarize_list)):
            summ=summarize_list[i]
            origin=origin_list[i]

            if i+1==answer: ## 단어가 바뀌는 경우
                new='('+str(i+1)+') '+new_list[i]
            else: 
                new='('+str(i+1)+') '+origin

            if origin in summ:
                new_summ=summ.replace(origin, new)
            else:
                new_summ=summ.lower().replace(origin, new)

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
        question_dict['question'] =self.question    ## '다음 글의 밑줄 친 부분 중, 문맥상 낱말의 쓰임이 적절하지 않은 것은?'

        summarize=self.summarize(passage)  ## list
        keyword=self.get_keyword(summarize) ## list
        keyword_antonym=self.get_keyword_antonym(keyword)   ## list

        ## 랜덤으로 답 뽑기
        ex=[1, 2, 3, 4, 5]
        answer = random.choice(ex)
        ex.remove(answer)

        # ans_list=keyword_antonym.copy() ## 틀린 단어(바뀌어야 함)
        # dist_list=keyword.copy()    ## 원래 단어(바뀌지 말아야 함)
        ans_list=keyword_antonym ## 틀린 단어(바뀌어야 함)
        dist_list=keyword    ## 원래 단어(바뀌지 말아야 함)

        ## 새로운 passage 만들기
        question_dict['new_passage']=self.make_new_passage(passage, summarize, dist_list, ans_list, answer)

        question_dict['answer']=str(answer)+'. '+ans_list[answer-1]
        question_dict['d1']=str(ex[0])+'. '+dist_list[ex[0]-1]
        question_dict['d2']=str(ex[1])+'. '+dist_list[ex[1]-1]
        question_dict['d3']=str(ex[2])+'. '+dist_list[ex[2]-1]
        question_dict['d4']=str(ex[3])+'. '+dist_list[ex[3]-1]

        return question_dict
#%%
#%% 38-39 문장이 들어가기에 적절한 곳
class Q6:
    def __init__(self):
        self.question_type=6
        self.question='글의 흐름으로 보아, 주어진 문장이 들어가기에 가장 적절한 곳을 고르시오.'
    
    def separate(self,passage:str):
        #문장단위로 쪼개기
        temp=passage.split('.') # 마침표 기준. 리스트로 쪼갬
        del temp[len(temp)-1]
        l=len(temp)

        # 정답 고르고 distractor 생성
        answer_list=[1,2,3,4,5]
        answer=random.randint(1,5)

        num=range(1,l-1)          # 문장 번호, 마지막 문장은 제외
        select=random.sample(num,5) # 그 중에 5개
        select.sort()

        dist_list=[x for x in answer_list if x!=answer]

        # 정답 문장
        ans_text=temp[select[answer-1]]

        # 일단 정답을 앞뒷문장이랑 합치고 
        temp[answer]=str(temp[answer-1])+'. ('+str(answer)+')'+str(temp[answer+1])
        del temp[answer-1]
        del temp[answer]
        del select[answer-1]

        # 나머지 문장에 괄호 붙히기
        cnt=0
        i=0
        l=len(temp)
        num=range(1,l)          
        select=random.sample(num,4) # 그 중에 4개
        select.sort()
        while(cnt<4):
            if(i==select[cnt]):
                if(i>=answer):
                    temp[i]='('+str(answer_list[cnt]+1)+')'+str(temp[i])
                else:
                    temp[i]='('+str(answer_list[cnt])+')'+str(temp[i])
                cnt+=1
            i+=1

        ## 문제 출력
        new_passage='. '.join(temp)
        new_passage=str(ans_text)+'\n\n'+str(new_passage)+'.'

        return new_passage, answer, dist_list

    def make_dict(self, passageID)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        new_passage, answer, dist_list=self.separate(passage)

        question_dict['new_passage'] = new_passage
        question_dict['answer']=answer
        question_dict['d1']=dist_list[0]   
        question_dict['d2']=dist_list[1]
        question_dict['d3']=dist_list[2]
        question_dict['d4']=dist_list[3]

        return question_dict

#%% 35 전체 흐름과 관계 없는 문장
class Q7:
    def __init__(self):
        self.question_type=8
        self.question='다음 글에서 전체 흐름과 관계 없는 문장은?'

    def summarize(self, passage:str, num_sentence=5)->list:
        return summary(passage, num_sentences=num_sentence)

    def get_completion_dict(self, summary:list)->dict:
        paraphrase=paraphrasing_by_transe(summary)  ## list
        sent_completion_dict=get_sentence_completions(paraphrase)#, option=Q7)   ## dict
        return sent_completion_dict

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

    def make_new_passage(self, passage:str, origin_list:list, false_list:list, answer:int)->str:
        new_passage=''+passage
        for i in range(len(origin_list)):
            origin=origin_list[i]

            if i+1==answer:
                new='('+str(i+1)+') '+false_list[i]
            else:
                new='('+str(i+1)+') '+origin
            
            if origin in passage:
                new_passage=new_passage.replace(origin, new)
            else:
                new_passage=new_passage.lower().replace(origin.lower(), new)

        return new_passage



    def make_dict(self, passageID)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question   ## 다음 글에서 전체 흐름과 관계 없는 문장은?

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################

        summarize=self.summarize(passage)  ## list
        sent_completion_dict=self.get_completion_dict(summarize)    ## dict
        false_paraphrase=self.get_false_paraphrase(sent_completion_dict)    ## list

        ## 랜덤으로 답 뽑기
        ex=[1, 2, 3, 4, 5]
        answer = random.choice(ex)
        ex.remove(answer)

        new_passage=self.make_new_passage(passage, summarize, false_paraphrase, answer)
        question_dict['new_passage']=new_passage
    
        question_dict['answer']=str(answer)
        question_dict['d1']=str(ex[0])
        question_dict['d2']=str(ex[1])
        question_dict['d3']=str(ex[2])
        question_dict['d4']=str(ex[3])

        return question_dict
#%% 지울것 !!!
def get_keyword_list(passage, max_word_cnt,top_n, option=None)->list:
    result=[]
    # option=None
    if type(passage)==list:
        for sentence in passage:
            kwd=kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1,max_word_cnt), stop_words='english', top_n=top_n)
            for i in range(len(kwd)):   ## 점수 빼고 단어만
                kwd[i]=kwd[i][0]
            if option==None:result.append(kwd)
            elif option=='Q8':result.append(kwd)
        # print(result)
        if option=='Q8':
            tmp=1;tmp_list=[]
            for a in result[0]:
                for b in result[1]:
                    similarity=word_similarity(a, b)
                    # print(a, b, similarity)
                    if similarity<tmp:
                        tmp=similarity; tmp_list=[a, b]  
            # print(tmp_list)
            result=tmp_list
                        
    elif type(passage)==str:
        kwd=kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, max_word_cnt), stop_words='english', top_n=top_n)
        for i in range(len(kwd)):   ## 점수 빼고 단어만
                kwd[i]=kwd[i][0]
        result=kwd

    return result
#%% 지우기 !!
class Pyphones:
    
    def __init__(self):
        self.url = "https://www.homophone.com/search?page={}&type=&q={}"
        
    def get_the_page(self, word, page_no=1 ):
        """
        Get the page content.
        Returns
            str: the content of the page.
        """
        url = self.url.format(page_no, word)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup

    def get_the_page_nos(self, word):
        """
        Get the total number of pages
        Returns
            int: the total number of the pages.
        """
        soup = self.get_the_page(word)
        pages = soup.find_all('div', attrs={'class':'col-sm-9'})
        total_pages = pages[0].find('h5').text.split('/')[-1].strip()
        return int(total_pages)

    def get_the_homophones(self, word):
        """
        Get the homophones of the word.
        Returns
            dict: {word: [list_of_homophones]} against each word.
        """
        total_pages = self.get_the_page_nos(word)
        for ix in range(total_pages):
            page_no = ix + 1
            soup = self.get_the_page(word, page_no)
            raw_homophones = soup.find_all('div', attrs={'class': 'well well-lg'})
            for elem in range(len(raw_homophones)):
                raw_homophones_2 = raw_homophones[elem].find_all('a', attrs={'class': 'btn word-btn'})
                list_of_homophones = list(raw_homophones_2)
                if any(list_of_homophones):
                    local_homophones = []
                    for tag_of_homophone in list_of_homophones:
                        homophone = tag_of_homophone.text
                        local_homophones.append(homophone)
        return local_homophones
#%% 40 글의 내용 요약하고 빈칸 2개 단어 고르기
## https://huggingface.co/docs/transformers/v4.21.0/en/main_classes/pipelines#transformers.SummarizationPipeline

class Q8:
    def __init__(self):
        self.question_type=8
        self.question='다음 글의 내용을 요약하고자 한다. 빈칸 (A), (B)에 들어갈 말로 가장 적절한 것은?'

    def summarize(self, passage:str, num_sentence=2)->list:
        return summary(passage, num_sentences=num_sentence)

    def paraphrase(self, summary:list)->list:
        return paraphrasing_by_transe(summary)

    def get_keyword(self, paraphrase:list, option='Q8') ->list:
        if option=='Q8':    ## option=='Q8' 인 경우 무조건 top_n==2
            top_n=2
        keyword_list= get_keyword_list(paraphrase, max_word_cnt=1, top_n=top_n, option=option)
        return keyword_list

    # def get_distractors(self, keyword:list)->list:    ## 오답 단어 4개 만들기
    #     wd=word_dict()
    #     py = Pyphones()

    #     distractors=[]
    #     for kwd in keyword:
    #         synonym_list=get_synonym_list(kwd, num_word=1)  ## 어려운 오답(유의어)
    #         homophones_list=py.get_the_homophones(kwd)
    #         print(synonym_list, homophones_list)
    #         # distractors=sum(antonym_list,[]) ## sum( 덧셈할 것, 처음에 더할 것)
    #         # distractors=sum(synonym_list,distractors) ## sum( 덧셈할 것, 처음에 더할 것)

    #     print(distractors)

    #     return distractors

    def make_new_passage(self, passage:str, paraphrase:list, keyword:list)->str:
        new_passage=passage+'\n\n==>'
        new_paraphrase=[]
        for i in range(len(paraphrase)):
            if keyword[i] in paraphrase[i]:
                if i==0:space='__(A)__'
                elif i==1:space='__(B)__'
                else:space='_____'  ## 이게 될 일은 없을걸...?
                new_paraphrase.append(paraphrase[i].replace(keyword[i],space))
        for sentence in new_paraphrase:
            new_passage=new_passage+' '+str(sentence)
        return new_passage

    def make_dict(self, passageID)->dict:
        question_dict=question_dict_sample.copy()
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=self.question_type
        question_dict['question'] = self.question

        #####################################################
        ## passageID로 passage를 가져오는 코드 있어야 함
        #####################################################
        
        summarize=self.summarize(passage)   ## list
        paraphrase=self.paraphrase(summarize)   ## list
        keyword=self.get_keyword(paraphrase)    ## list
        new_passage=self.make_new_passage(passage, paraphrase, keyword)
        distractors=self.get_distractors(keyword)

        question_dict['new_passage']=new_passage

        question_dict['answer']='(A)'+keyword[0]+' (B)'+keyword[1]

        # question_dict['d1']
        # question_dict['d2']
        # question_dict['d3']
        # question_dict['d4']

        return question_dict
#%%
q8=Q8()
q8_dict=q8.make_dict(2)
print(q8_dict)
# print(q8.make_dict(2))

# %%

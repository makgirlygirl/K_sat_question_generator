#%% import package
import random
import warnings

from transformers import logging

from K_sat_question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore') 

#%% file open & set seed(for same result)
#####################################################
## K_sat_function 에서 make_dict 에 
## DB에서 passageID로 passage를 가져오는 코드 있어야 함
#####################################################
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()

# random.seed(1018)## 나랑 영재 만난 날 이지롱 >_<
random.seed(42)
#%% 18, 20, 22(목적/요지/주제): 한국어 보기, 23, 24, 41(주제/제목): 영어 보기
q1=Q1()
q1_dicr_eng=q1.make_dict(passageID, is_Korean=False)
print(q1_dicr_eng)
q1_dicr_kor=q1.make_dict(passageID, is_Korean=True)
print(q1_dicr_kor)
#%% 26-28, 45(내용 일치/불일치): 영어 보기/한국어보기
q2=Q2()
q2_dict_eng=q2.make_dict(passageID, is_Korean=False)
print(q2_dict_eng)
q2_dict_kor=q2.make_dict(passageID, is_Korean=True)
print(q2_dict_kor)
#%% 36-37, 43(순서(ABC)): 영어 보기
q3=Q3()
q3_dict=q3.make_dict(passageID)
print(q3_dict)
#%% 31(빈칸추론(단어)): 영어 보기
q4=Q4()
q4_dict=q4.make_dict(passageID)
print(q4_dict)
#%% 30, 42(적절하지 않은 단어)
q5=Q5()
q5_dict=q5.make_dict(passageID)
print(q5_dict)
#%% 38-39 문장이 들어가기에 적절한 곳
q6=Q6()
q6_dict=q6.make_dict(passageID)
print(q6_dict)
#%%

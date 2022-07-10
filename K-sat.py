#%% import package
import random
import warnings

from transformers import logging

from K_sat_question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')

#%% file open & set seed(for same result)

f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()

random.seed(1018)## 나랑 영재 만난 날 이지롱 >_<
#%%
# 18, 20, 22(목적/요지/주제): 한국어 보기
# 23, 24, 41(주제/제목): 영어 보기
q1=Q1()
q1_question_type=1
q1_list=['목적으로', '주장으로', '요지로']## 제목이 좀 어색하긴 함
q1_question=[f'다음 글의 {random.choice(q1_list)} 가장 적절한 것은?']
q1_summarize=q1.summarize(passage)
q1_paraphrase, q1_sent_completion_dict=q1.paraphrase(q1_summarize)
q1_distractors=q1.distractors(q1_sent_completion_dict)
q1_dict_kor=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors,is_Korean=True)
q1_dict_eng=q1.make_dict(passageID, q1_question_type, q1_question, q1_paraphrase, q1_distractors)


# %%
# 26-28, 45(내용 일치/불일치): 영어 보기

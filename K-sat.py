#%% import package
import random
import warnings

from transformers import logging

# from function import get_sentence_with_ans, pick_question, transe, transe_kor
# from K_sat_function import summary
from K_sat_question import *

# from question import *

logging.set_verbosity_warning()
warnings.filterwarnings(action='ignore')

#%% file open & set seed(for same result)

f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()

random.seed(1018)## 나랑 영재 만난 날 이지롱 >_<
#%% 18(목적), 20(주장), 22(요지)
q1=Q1()
q1_question_type=1
q1_list=['목적으로', '주장으로', '요지로']
q1_question=[f'다음 글의 {random.choice(q1_list)} 가장 적절한 것은?']
print(q1_question)

q1_summarize=q1.summarize(passage)
print(q1_summarize)

#%%
q1_paraphrase=q1.paraphrase(q1_summarize)   
print(q1_paraphrase)
#%%

q1_distractors=q1.distractors(q1_paraphrase)
#%%    
q1_K_distractors=[]
for sentence in q1_distractors:
    q1_K_distractors.append(transe_kor(sentence))
#%%
q1_dict=q1.make_dict(passageID, q1_paraphrase, q1_distractors)


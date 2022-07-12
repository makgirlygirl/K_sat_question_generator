#%%
from transformers import logging

logging.set_verbosity_warning()
import itertools
import warnings

import nltk
import torch

warnings.filterwarnings(action='ignore')
# nltk.download('stopwords')
from nltk.corpus import stopwords

from function import (benepar_parser, encode_qa_pairs, generate_distractor,
                      generate_sentences, get_keyword,
                      get_sentence_completions, get_sentence_with_ans,
                      preprocess, sent_tokenize, transe, transe_kor, word_dict,
                      word_similarity)
from model import (bert_model, gpt2_model, gpt2_tokenizer, paraphrase_model,
                   paraphrase_tokenizer, qae_model, qae_tokenizer, qg_model,
                   qg_tokenizer, summarize_model, summarize_tokenizer,
                   unmasker)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
question_dict={'passageID':None,
                'question_type':None,
                'question':None, 
                'answer':None,
                'd1':None, 'd2':None, 'd3':None, 'd4':None}
#%%
## MCQ 쓸 유형별로 조금씩 변경해서 MCQ1, MCQ2, ...등 만들기->question_type에 이걸 넣을것
class MCQ:
    def __init__(self):
        self.summarize_tokenizer=summarize_tokenizer
        self.paraphrase_tokenizer = paraphrase_tokenizer
        self.gpt2_tokenizer = gpt2_tokenizer

        self.summarize_model = summarize_model
        self.paraphrase_model = paraphrase_model
        self.gpt2_model = gpt2_model
        self.bert_model=bert_model
    
    # def summarize(self, passageID):
        # passageID로 passage를 가져온다(db랑 연결이 되어야 할 듯 ??)
    # 일단은 passage로 함수를 만들어봄
    def summarize(self, passage, isTranse=True):
        ## passageID로 passage를 가져온다(db랑 연결이 되어야 할 듯 ??)
        inputs = summarize_tokenizer.encode("summarize: " + passage, return_tensors="pt", max_length=2000)
        inputs = inputs.to(DEVICE)
        outputs = summarize_model.generate(
            inputs,
            max_length=1000,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True)
            
        extracted_sentences = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        tokenized_sentences = nltk.tokenize.sent_tokenize(extracted_sentences)

        ## 중복되는 문장 나오지 않게
        filter_quotes_and_questions = list(set(preprocess(tokenized_sentences, isTranse)))
        return filter_quotes_and_questions# list
    
    ## summarize의 결과가 paraphrase의 input으로 들어감
    def paraphrase(self, filter_quotes_and_questions, isTranse=True):# list->list
        paraphrased_sentences=[]
        for summary_idx in range(len(filter_quotes_and_questions)):
            sentence = filter_quotes_and_questions[summary_idx]
            inputs = "paraphrase: " + sentence + " </s>"

            encoding = paraphrase_tokenizer.encode_plus(inputs, pad_to_max_length=True, return_tensors="pt")
            input_ids, attention_masks = encoding["input_ids"].to(DEVICE), encoding["attention_mask"].to(DEVICE)
            num_return_sequences=len(filter_quotes_and_questions)

            outputs = paraphrase_model.generate(
                input_ids=input_ids, attention_mask=attention_masks,
                max_length=1000,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                early_stopping=True,
                num_return_sequences=4
                )

            if isTranse==True:
                paraphrased_sentences.append(transe(paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)))
            else:
                paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True))
            
            if len(paraphrased_sentences) == 4:
                break

            # 마지막인데 채워지지 않았을 경우 존재하는 paraphrased sentence 반복해서 false 문장 생성
            if (summary_idx == (num_return_sequences - 1)) & (len(paraphrased_sentences) < 4): 
                for paraphrase_idx in range(1, 5):
                    if isTranse==True:
                        paraphrased_sentences.append(transe(paraphrase_tokenizer.decode(outputs[paraphrase_idx], skip_special_tokens=True,clean_up_tokenization_spaces=True)))
                    else:
                        paraphrased_sentences.append(paraphrase_tokenizer.decode(outputs[paraphrase_idx], skip_special_tokens=True,clean_up_tokenization_spaces=True))
            
        sent_completion_dict=get_sentence_completions(paraphrased_sentences)
        return sent_completion_dict

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def distractors(self, sent_completion_dict, isTranse=True):
        distractors=[]
        distractor_cnt = 1

        for key_sentence in sent_completion_dict:
            if distractor_cnt == 5:
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
            # print(false_sentences)
            distractors.append(transe(false_sentences[0]))
            distractor_cnt += 1
        return distractors

    ## paraphrase 의 결과가  complete_dict의 input으로 들어감
    def make_dict(self, passageID,question_type, question, sent_completion_dict, false_sentences):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']=question_type
        question_dict['question'] = question
        question_dict['answer']=sent_completion_dict
        question_dict['d1']=false_sentences[0]
        question_dict['d2']=false_sentences[1]
        question_dict['d3']=false_sentences[2]
        question_dict['d4']=false_sentences[3]

        return question_dict

# # %%
'''class WH:
    def __init__(self):
        self.qg_tokenizer=qg_tokenizer
        self.qae_tokenizer = qae_tokenizer

        self.qg_model = qg_model
        self.qae_model = qae_model

        self.unmasker = unmasker

    
        def answers_generate(self, passage):
            answers=get_NP(passage)
            print(answers)
            answers=ans_len_limit(answers)
            print(answers)

            stop_words = set(stopwords.words('english'))
            answers=remove_stopwords(stop_words, answers)
            print(answers)

            return answers
        
    ## 정답 생성
    def answers_generate(self, passage):
        answers=get_keyword(passage)
        pair=list(itertools.combinations((answers),2))
        for i in pair:
            a=i[0]; b=i[1]
            if word_similarity(a, b)>0.8:
                if b in answers:
                    answers.remove(b)
        return answers
    ## get_NP의 결과가 question_generate의 인풋으로 들어감
    ## 문제 생성
    def question_generate(self, passage, answers):
        ANSWER_TOKEN = "<answer>"
        CONTEXT_TOKEN = "<context>"
        SEQ_LENGTH = 512## 2000이었나? 아무튼 바꾸기(function.py의 encode_qa_pairs 에도 있음)
        
        questions = []

        for ans in answers: 
            qg_input = "{} {} {} {}".format(ANSWER_TOKEN, ans, CONTEXT_TOKEN, passage)
                
            encoded_input =qg_tokenizer(qg_input, padding='max_length', max_length=SEQ_LENGTH, truncation=True, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                output =qg_model.generate(input_ids=encoded_input["input_ids"])
            question = qg_tokenizer.decode(output[0], skip_special_tokens=True)
            questions.append(question)
        return questions

    ## question_generate, get_NP의 결과가 인풋으로 들어감
    ## 문제 평가
    def get_scores(self, questions, answers):
        encoded_qa_pairs=encode_qa_pairs(questions, answers)## list: 인코딩
        scores = {}
        qae_model.eval()
        with torch.no_grad():
            for i in range(len(encoded_qa_pairs)):
                scores[i] = self.qae_model(**encoded_qa_pairs[i])[0][0][1]
        return [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)]

    def get_synonym_dict(self, word):
        dictionary=word_dict(word.split())
        synonym= (dictionary.getSynonyms())
        return synonym
    
    def get_antonym_dict(self, word):
        dictionary=word_dict(word.split())
        antonym= (dictionary.getAntonyms())
        return antonym
    
    ## 오답생성
    def distractors(self, passage, answer):
        distractors = []
        sentences = nltk.sent_tokenize(passage)
        NN=get_NN(answer)
        target_sentence=get_sentence_with_ans(sentences, answer)
        for i in range(4):
            print(generate_distractor(target_sentence, 9-i, answer, NN))
            print(transe(generate_distractor(target_sentence, 9-i, answer, NN)))
            distractors.append(generate_distractor(target_sentence, 9-i, answer, NN))
    
    ## dict 리턴
    def make_dict(self, passageID, answers, question, distractors):
        question_dict['passageID']=int(passageID)
        question_dict['question_type']='WH'
        question_dict['question'] = question.split("?")[0]+"?"
        # question_dict['answer']=answers
        # question_dict['d1']=distractors[0]
        # question_dict['d2']=distractors[1]
        # question_dict['d3']=distractors[2]
        # question_dict['d4']=distractors[3]

        # distractors = []
        # for i in range(3):
        #     distractors.append(generate_distractor(target_sentence, 9-i, answers[index], NNs))
    

        return question_dict


# %%
'''

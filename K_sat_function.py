#%% impotr package

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
import re
from difflib import SequenceMatcher
from string import punctuation
from typing import Dict, List
from urllib.request import urlopen

import benepar
import requests
import scipy
from bs4 import BeautifulSoup
from nltk import tokenize
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words

benepar_parser = benepar.Parser("benepar_en3")

import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%
from model import bert_model, gpt2_model, gpt2_tokenizer, kw_model, translator

#%%
f = open("/home/my_qg/testset/2.txt","r")
passageID=2
passage = f.read()

#%%
def summary(passage, num_sentences):
    parser = PlaintextParser.from_string(passage, Tokenizer('english'))
    stemmer = Stemmer('english')
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words('english')

    summary=[]

    for sentence in summarizer(parser.document, num_sentences):
        summary.append(str(sentence))
    return summary

def paraphrasing_by_transe(summary, midpoint='zh-cn')->list:
    if type(summary)==list:
        pharaphrase=[]
        for sentence in summary:
            translate=translator.translate(sentence, src='en', dest=midpoint).text
            pharaphrase.append(translator.translate(translate, src=midpoint, dest='en').text)
        return pharaphrase
    elif type(summary)==str:
        translate=translator.translate(summary, src='en', dest=midpoint).text
        pharaphrase=''+translator.translate(translate, src=midpoint, dest='en').text
        return pharaphrase
    else:
        print('paraphrasing_by_transe: input error(input type must be list or str)')
        print('your input is'+type(sentence))
        return None

def transe_kor(sentence):
    # print(type(sentence))
    if type(sentence)==list:
        kor=[]
        for sent in sentence:
            k_sentence=translator.translate(sent, src='en', dest='ko').text
            kor.append(k_sentence)
        return kor
    elif type(sentence)==str:
        k_sentence=translator.translate(sentence, src='en', dest='ko').text
        kor=''+k_sentence
        return kor
    else :
        print('transe_kor: input error(input type must be list or str)')
        print('your input is'+type(sentence))
        return None
#%%
def sort_by_similarity(original_sentence, generated_sentences_list):

    sentence_embeddings = bert_model.encode(generated_sentences_list)
    queries = [original_sentence]
    query_embeddings = bert_model.encode(queries)
    number_top_matches = len(generated_sentences_list)
    dissimilar_sentences = []

    for query, query_embedding in zip(queries, query_embeddings):
        distances = scipy.spatial.distance.cdist([query_embedding], sentence_embeddings, "cosine")[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        for idx, distance in reversed(results[0:number_top_matches]):
            score = 1-distance
            # print(score)
            if score < 0.99:
                dissimilar_sentences.append(generated_sentences_list[idx].strip())
           
    sorted_dissimilar_sentences = sorted(dissimilar_sentences, key=len)
    return sorted_dissimilar_sentences[:2]

def generate_sentences(partial_sentence,full_sentence):
    input_ids = gpt2_tokenizer.encode(partial_sentence, return_tensors='pt') # use tokenizer to encode
    input_ids = input_ids.to(DEVICE)
    maximum_length = len(partial_sentence.split())+80 

    sample_outputs = gpt2_model.generate( 
        input_ids,
        do_sample=True,
        max_length=maximum_length, 
        top_p=0.90, 
        top_k=50,   
        repetition_penalty  = 10.0,
        num_return_sequences=5
    )
    generated_sentences=[]
    for i, sample_output in enumerate(sample_outputs):
        decoded_sentences = gpt2_tokenizer.decode(sample_output, skip_special_tokens=True)
        decoded_sentences_list =tokenize.sent_tokenize(decoded_sentences)
        generated_sentences.append(decoded_sentences_list[0]) # takes the first sentence 
        
    top_3_sentences = sort_by_similarity(full_sentence, generated_sentences)
    return top_3_sentences
#%%
def preprocess(sentences):
    output = []
    for sent in sentences:
        single_quotes_present = len(re.findall(r"['][\w\s.:;,!?\\-]+[']",sent))>0
        double_quotes_present = len(re.findall(r'["][\w\s.:;,!?\\-]+["]',sent))>0
        question_present = "?" in sent
        if single_quotes_present or double_quotes_present or question_present :
            continue
            print(type(sent.strip(punctuation)))
        else:
            output.append(sent.strip(punctuation))
    return output
def get_flattened(t):# MCQ, WH
    sent_str_final = None
    if t is not None:
        sent_str = [" ".join(x.leaves()) for x in list(t)]
        sent_str_final = [" ".join(sent_str)]
        sent_str_final = sent_str_final[0]
    return sent_str_final

def get_termination_portion(main_string,sub_string):
    combined_sub_string = sub_string.replace(" ","")
    main_string_list = main_string.split()
    last_index = len(main_string_list)
    for i in range(last_index):
        check_string_list = main_string_list[i:]
        check_string = "".join(check_string_list)
        check_string = check_string.replace(" ","")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])       
    return None

def get_right_most_VP_or_NP(parse_tree,last_NP = None,last_VP = None):
    if len(parse_tree.leaves()) == 1:
        return get_flattened(last_NP),get_flattened(last_VP)
    last_subtree = parse_tree[-1]
    if last_subtree.label() == "NP":
        last_NP = last_subtree
    elif last_subtree.label() == "VP":
        last_VP = last_subtree
    
    return get_right_most_VP_or_NP(last_subtree,last_NP,last_VP)

def get_sentence_completions(key_sentences, option=None):
    sentence_completion_dict = {}
    for individual_sentence in key_sentences:
        sentence = individual_sentence.rstrip('?:!.,;')
        tree = benepar_parser.parse(sentence)
        last_nounphrase, last_verbphrase =  get_right_most_VP_or_NP(tree)
        phrases= []
        if last_verbphrase is not None:
            verbphrase_string = get_termination_portion(sentence,last_verbphrase)
            if verbphrase_string is not None:
                phrases.append(verbphrase_string)
                
        if last_nounphrase is not None:
            nounphrase_string = get_termination_portion(sentence,last_nounphrase)
            if nounphrase_string is not None:
                phrases.append(nounphrase_string)
    
        longest_phrase =  sorted(phrases, key=len, reverse=True)
        if len(longest_phrase) == 2:
            first_sent_len = len(longest_phrase[0].split())
            second_sentence_len = len(longest_phrase[1].split())
            if (first_sent_len - second_sentence_len) > 4:
                if option==None:
                    del longest_phrase[1]
                elif option=='Q7':
                    del longest_phrase[0]
                
        if len(longest_phrase)>0:
            sentence_completion_dict[sentence]=longest_phrase

    return sentence_completion_dict
## %%
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


def word_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()
#%%
def _get_soup_object(url, parser="html.parser"):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.84 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
    }
    return BeautifulSoup(requests.get(url,headers=headers).text, parser)

class word_dict(object):
    def __init__(self, *args):
        try:
            if isinstance(args[0], list):
                self.args = args[0]
            else:
                self.args = args
        except:
            self.args = args
    
    def getSynonyms(self, num_word, formatted=False):
        return [self.synonym(term, num_word, formatted) for term in self.args]
        
    def getAntonyms(self,num_word, formatted=False):
            return [self.antonym(term, num_word, formatted) for term in self.args]

    def synonym(self, term, num_word, formatted=False):
        if len(term.split()) > 1:
            print("Error: A Term must be only a single word")
        else:
            try:
                data = _get_soup_object("https://www.powerthesaurus.org/"+term+"/synonyms")
                ## 여기 클래스가 가끔 바뀌는 듯..?
                # section = data.findAll('a', {'class': "ch_at ch_ci aaa_at"})[:num_word]
                section = data.findAll('a', {'class': "cl_az cl_cm z4_az"})[:num_word]
                synonyms=[s.text.strip() for s in section]
                # print(synonyms)
                if formatted:
                    return {term: synonyms}
                return synonyms
            except: None

    def antonym(self, term, num_word, formatted=False):
        if len(term.split()) > 1:
            print(term)
            print("Error: A Term must be only a single word")
        else:
            try:
                data = _get_soup_object("https://www.powerthesaurus.org/"+term+"/antonyms")
                ## 여기 클래스가 가끔 바뀌는 듯..?
                # section = data.findAll('a', {'class': "ch_at ch_ci aaa_at"})[:num_word]
                section = data.findAll('a', {'class': "cl_az cl_cm z4_az"})[:num_word]
                antonyms=[s.text.strip() for s in section]
                if formatted:
                    return {term: antonyms}
                return antonyms
            except:None

def get_synonym_list(word:str, num_word)->list:
    dictionary=word_dict(word.split())
    synonym= dictionary.getSynonyms(num_word=num_word)
    return synonym

def get_antonym_list(word:str, num_word)->list:
    dictionary=word_dict(word.split())
    antonym= dictionary.getAntonyms(num_word=num_word)
    return antonym
#%%
"""
Python wrapper for the website: https://www.homophone.com/
Gets the homophones of a word.
https://github.com/kjanjua26/Pyphones
"""
class Pyphones:
    
    def __init__(self):
        # self.word = word
        self.url = "https://www.homophone.com/search?page={}&type=&q={}"
        # self.homophones = {self.word: []}
        
    def get_the_page(self, page_no=1):
        """
        Get the page content.
        Returns
            str: the content of the page.
        """
        url = self.url.format(page_no, self.word)
        r = requests.get(url)
        soup = BeautifulSoup(r.content, "html.parser")
        return soup

    def get_the_page_nos(self):
        """
        Get the total number of pages
        Returns
            int: the total number of the pages.
        """
        soup = self.get_the_page()
        pages = soup.find_all('div', attrs={'class':'col-sm-9'})
        total_pages = pages[0].find('h5').text.split('/')[-1].strip()
        return int(total_pages)

    def get_the_homophones(self, word):
        """
        Get the homophones of the word.
        Returns
            dict: {word: [list_of_homophones]} against each word.
        """
        total_pages = self.get_the_page_nos()
        for ix in range(total_pages):
            page_no = ix + 1
            soup = self.get_the_page(page_no)
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
        #             self.homophones[self.word].append(local_homophones)

        # return self.homophones

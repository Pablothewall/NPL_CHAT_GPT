# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:14:18 2023

@author: l11420
"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')



def lemma(text):
    text=str(text)
    lemmatizer=WordNetLemmatizer()
    return lemmatizer.lemmatize(text)


def join_lists_to_string(lst):
    return ' '.join(lst)




nltk.download('wordnet')

stop_words = set(stopwords.words('english')) 
print(stop_words)
# Define a function to remove stop words from a sentence 
def remove_stop_lemma_words(sentence): 
  words = sentence.split() 
  palabras= len(words)
  filtered_words = [str.lower(word) for word in words if str.lower(word) not in stop_words] 
  lemmatized_words=[]
  lemma_1=0
  no_lemma=0
  for word in filtered_words:
      
      if lemma(word) is not None:
          lemmatized_words.append(lemma(word))
          lemma_1+=1
      else:
          lemmatized_words.append(word)
          no_lemma+=1    


  palabras_2=len(lemmatized_words)
  print(palabras_2-palabras)

  return ' '.join(filtered_words)




def tokenIze(text):
    words = word_tokenize(text)
    return words
    
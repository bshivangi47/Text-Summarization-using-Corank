# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:15:10 2020

@author: SONY
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 13:29:48 2019

@author: SONY
"""

import time
t0= time.time()
import nltk  
import re 
#from nltk.corpus import stopwords 
#from collections import Counter
import glob
import numpy as np
#
#
# Reading text file
#
#

text1=[]

'''list_of_folders = glob.glob("C:/Users/SONY/Desktop/testdocs/*") 
for folder in list_of_folders:
    f=folder+"/*"
    files_in_folders = glob.glob(f) 
    text=""
    for file in files_in_folders:
        file1 = open(file,"r+")
        text+="\n"+file1.read()
    text1.append(text)
    
text1=np.asarray(text1) '''

list_of_folders = glob.glob("C:/Users/SONY/Desktop/preprocessed_docs/*") 
for (folder,i) in zip(list_of_folders,range(len(list_of_folders))):
    f=folder+"/*"
    files_in_folders = glob.glob(f) 
    text=""
    for file in files_in_folders:
        file1 = open(file,"r+",encoding = "ISO-8859-1")
        text+="\n"+file1.read()
        file1.close() 
    text1.append(text) 
#delete_list = ["<P>","</P>","</TEXT>", "</BODY>","<BODY>","<TEXT>", "</DOC>","<DOC>"]
   

def remove_between_square_brackets(text):
    text = re.sub('</P>','',text)
    text = re.sub('<DOCNO>.*</DOCNO>','',text)
    text= re.sub('<DOCTYPE>.*</DOCTYPE>','',text)
    text = re.sub('<BODY>','',text)
    text = re.sub('</BODY>','',text)
    text = re.sub('<TEXT>','',text)
    text= re.sub('</P>','',text)
    text= re.sub('<P>','',text)
    text= re.sub('</DOC','',text)
    text= re.sub('<DOC>','',text)
    text= re.sub('</TEXT>','',text)
    text= re.sub('<DATE_TIME>.*</DATE_TIME>','',text)
    text= re.sub('<HEADLINE>.*</HEADLINE>','',text)
    text= re.sub('<HEADER>.*</HEADER>','',text)
    text= re.sub('<TRAILER>.*</TRAILER>','',text)
    text= re.sub('<SLUG>.*</SLUG>','',text)
    text= re.sub('<CATEGORY>.*</CATEGORY>','',text)
    return re.sub('\[[^]]*\]', '', text)
    
#def remove_brackets(text):
 #   for line in text:
  #      for word in delete_list:
   #         return re.sub(word, '', text)
        
def denoise_text(text):
    text = remove_between_square_brackets(text)
    #text = remove_brackets(text)
    return text  

#
#
#Preprocessing
#
#
#P is a vector of all the sentences

def summarize_text(article_text,n):
    #article_text = denoise_text(article_text)
    #delete_list = ["<P>","</P>","</TEXT>", "</BODY>","<BODY>","<TEXT>", "</DOC>","<DOC>"]
    P = nltk.sent_tokenize(article_text)
    corpus=nltk.sent_tokenize(article_text)
    
    '''stop_words = set(stopwords.words("english"))
    stopwords_dict = Counter(stop_words)
    for k in range(len(corpus )):
        corpus [k] = re.sub('<DOCNO>.*</DOCNO>','',corpus[k])
        corpus[k]= re.sub('<DOCTYPE>.*</DOCTYPE>','',corpus[k])
        corpus[k] = re.sub('<BODY>','',corpus[k])
        corpus[k] = re.sub('</BODY>','',corpus[k])
        corpus[k] = re.sub('<TEXT>','',corpus[k])
        corpus[k] = re.sub('</P>','',corpus[k])
        corpus[k]= re.sub('<P>','',corpus[k])
        corpus[k]= re.sub('</DOC','',corpus[k])
        corpus[k]= re.sub('<DOC>','',corpus[k])
        corpus[k]= re.sub('</TEXT>','',corpus[k])
        corpus[k]= re.sub('<DATE_TIME>.*</DATE_TIME>','',corpus[k])
        corpus[k]= re.sub('<HEADLINE>.*</HEADLINE>','',corpus[k])
        corpus[k]= re.sub('<HEADER>.*</HEADER>','',corpus[k])
        corpus[k]= re.sub('<TRAILER>.*</TRAILER>','',corpus[k])
        corpus[k]= re.sub('<SLUG>.*</SLUG>','',corpus[k])
        corpus[k]= re.sub('<CATEGORY>.*</CATEGORY>','',corpus[k])
        #word_tokens = nltk.word_tokenize(corpus [k]) 
        corpus [k] = ' '.join([word for word in corpus[k].split() if word not in stopwords_dict])
       '''
#
#        
# Word frequency list
#
#
    #corpus=P
    for k in range(len(corpus )):  
          corpus [k] = corpus [k].lower()
          corpus [k] = re.sub(r'\W',' ',corpus [k])
          corpus [k] = re.sub(r'\s+',' ',corpus [k])
    wordfreq = {}
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
#
#
#bag_of words = frequency of jth word in ith sentence
#
# 
           
    bag_of_words = []
    for sentence in corpus:
        sentence_tokens = nltk.word_tokenize(sentence)
        sent_vec = []
        for token in wordfreq:
            if token in sentence_tokens:
                sent_vec.append(sentence_tokens.count(token))
            else:
                sent_vec.append(sentence_tokens.count(token))
        
        bag_of_words.append(sent_vec)


    bag_of_words = np.asarray(bag_of_words)

#
#
# adjacency matrix = jaccard similarity between two matrices
#
#


    adjacency_matrix=[]
    for sentence in corpus:
        s1=sentence
        adjacency_count=[]
        for sentence in corpus:
            s2=sentence
            adjacency_count.append(1-nltk.jaccard_distance(set([s1]),set([s2])))
        adjacency_matrix.append(adjacency_count)
    adjacency_matrix=np.asarray(adjacency_matrix)
            
#
#
#Matrix_B= row normalisation of bag of words model
#
#

    matrix_B = np.array([[0 for x in range(len(bag_of_words[0]))] for y in range(len(bag_of_words))])

    sum=[]
    for i in range(len(bag_of_words)):
        total=0
        for j in range(len(bag_of_words[0])):
            total=total+bag_of_words[i][j]
        sum.append(total)

    sum=np.array(sum)

    for i in range(len(bag_of_words)):
        for j in range(len(bag_of_words[0])):       
            matrix_B[i][j]= np.nan_to_num(bag_of_words[i][j]/sum[i])
        

    matrix_B=np.asarray(matrix_B)

#
#
# scoring sentences
#
#
    sentence_score_t_1 = np.random.rand(len(corpus))
#sentence_score_t_1 = np.zeros(np.asarray(corpus).shape, dtype=float, order='C')
    alpha=0.6
    epsilon=0.00001
    first_term=np.multiply(alpha,adjacency_matrix)
    B_transpose=np.transpose(matrix_B)
    temp=np.matmul(matrix_B,B_transpose)
    second_term=np.multiply(1-alpha,temp)
    constant_matrix=np.add(first_term,second_term)
    while True:
        sentence_score_t= constant_matrix.dot(sentence_score_t_1)
        condition = np.linalg.norm(np.subtract(sentence_score_t,sentence_score_t_1))
        if condition*condition > epsilon:
            break
        else:
            sentence_score_t_1=sentence_score_t
       
#print(sentence_score_t)    
    output=""
    
    index = (sentence_score_t).argsort()[:n]
    index=np.sort(index)
    for i in range(len(index)):
        t=index[i]
        output+=P[t]+"\n"
    return output
    
     
def word_limit(text):
    if(len(text.split())>250):
        return ' '.join(text.split()[:250])
    
    


n=int(input("Enter the size of summary"))
for i in range(len(text1)):
    string= "C:/Users/SONY/Desktop/summary/summary"+str(i)+".txt"
    file2 = open(string,"w") 
    output=word_limit(summarize_text(text1[i],n))
    print(output)
    file2.write(output)
    file2.close()
    print("***********************************************************************************")

    
print("done")
t1 = time.time()
print("Time elapsed: ", t1 - t0)       

#file2.write(output)
#file2.close()
print("done")


       
    

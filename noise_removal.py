# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 11:28:22 2020

@author: SONY
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 19:26:05 2020

@author: SONY
"""

import glob
import numpy as np
from bs4 import BeautifulSoup
#
#
# Reading text file
#
#

text1=[]

def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 

list_of_folders = glob.glob("C:/Users/SONY/Desktop/testdocs/*") 
for folder in list_of_folders:
    f=folder+"/*"
    files_in_folders = glob.glob(f) 
    text=""
    for file in files_in_folders:
        file1 = open(file,"r+")
        text+="/n"+file1.read()
    text1.append(text)
    
text1=np.asarray(text1)   


def preprocessing(text):
    soup = BeautifulSoup(text, "html.parser")
    article=listToString([p.get_text() for p in soup.find_all("p", text=True)])
    article = article.strip()
    return article

   
list_of_folders = glob.glob("C:/Users/SONY/Desktop/preprocessed_docs/*") 
for (folder,i) in zip(list_of_folders,range(len(list_of_folders))):
    f=folder+"/*"
    files_in_folders = glob.glob(f) 
    text=""
    for file in files_in_folders:
        file2 = open(file,"r+",encoding = "ISO-8859-1")
        file2.write(preprocessing(text1[i]))
        file2.close()
        
print("!!!!!!")    


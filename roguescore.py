# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 17:47:57 2020

@author: SONY
"""

from rouge import Rouge
import glob


def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 

list_of_files1 = glob.glob("C:/Users/SONY/Desktop/summary/*") 
list_of_files2 = glob.glob("C:/Users/SONY/Desktop/ref_summary/*") 

with open("C:/Users/SONY/Desktop/scores.txt",'w') as data:
    
    for (sys_file,ref_file,i) in zip(list_of_files1,list_of_files2,range(len(list_of_files1))):
        file1 = open(sys_file,"r+")
        file2 = open(ref_file,"r+")
        ref = file2.read()
        sys = file1.read()
        r = Rouge()
        scores = r.get_scores([sys], [ref])
        print(scores)
        data.write("summary_"+str(i)+":"+"\n"+"\t"+str(scores)+"\n"+"***********************************************************************************"+"\n")
        print("***********************************************************************************")
        #file3.write(listToString(scores))
 
#file3.close()
print("done")
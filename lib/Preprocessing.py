import pandas as pd
from ast import literal_eval
import os
from lib.functions import  number_ascii_inseq,number_non_zero,similarity_index
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def list_files(dir):                                                                                                  
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):                                                                                          
            for file in files:
                r.append(os.path.join(subdir, file))                   
    return r

class preprocess:
    def __init__(self,dir_path,file_nbr):
        self.df = None
        self.file_nbr = file_nbr   
        self.files =[]
        self.seq_len=0
        self.load(dir_path)
        
        
        print("#### Done ####")
        
    def len_vocab(self):
        return len(self.vocab)
    
    def load(self,dir_path):
        print('#### Loading files ####')
        files=list_files(dir_path)
        files.sort()
        files=files[:self.file_nbr]
        dataframes=[]
        req_cols=["hex_values","classes","Label"]
        for file in files:
            self.files.append(file)
            print(file)
            df = pd.read_csv(file,sep='\t',usecols=req_cols)
            df['hex_values'] = df['hex_values'].apply(literal_eval)
            df['classes'] = df['classes'].apply(literal_eval)
            dataframes.append(df)
        data=pd.concat(dataframes,ignore_index=True)
        self.df = data
        self.seq_len=len(self.df['hex_values'][0])
        temp=[]
        for idx, value in self.df['classes'].items():
            try:
                temp.append(value.index(1))
            except:
                temp.append(self.seq_len)
        self.df['Position'] = pd.Series(temp)
        self.df.drop(columns='classes', inplace=True)
        


          
    def build_vocab(self):
        self.vocab={}
        print('#### Building vocab ####')     
        i=1
        for idx, value in self.x.iteritems():
            for element in value:
                if element in self.vocab:
                    pass
                else:
                    self.vocab[element]=i
                    i=i+1
            
    def vectorize(self):
        print('#### vectorization ####')
        for idx, value in self.x.iteritems():
            for i in range(len(value)):
                try:
                    value[i]=self.vocab[value[i]]
                except:
                    value[i]=0
                    
        
class preprocess_meta_data:
    def __init__(self,dir_path,file_nbr,threshhold, permutations):
        self.threshhold = threshhold
        self.permutations = permutations
        self.x=None
        self.y = None
        self.df = None
        self.file_nbr = file_nbr
        self.load(dir_path)
        self.add_meta()
        print('############ Done ############')
        
    def class_count(self):
        count_p = self.df[self.df['Label'] == 1].shape[0]
        count_n = self.df[self.df['Label'] ==0].shape[0]
        print('Positive class count : {}, Negative class count : {}'.format(count_p, count_n))
    
    def load(self,dir_path):
        print('############ Loading Data ############')
        files=list_files(dir_path)
        files.sort()
        files=files[:self.file_nbr]
        dataframes=[]
        req_cols=['values',"Label"]
        for file in files:
            print(file)
            df = pd.read_csv(file,sep='\t',usecols=req_cols)
            df['values'] = df['values'].apply(literal_eval)
            dataframes.append(df)
        data=pd.concat(dataframes,ignore_index=True)
        self.df = data
        

    def add_meta(self):
        print('############ Adding metadata ############')
        self.df['non_zero_word_count'] = self.df['values'].apply(lambda x: number_non_zero(x) )
        self.df['asci_count_inseq'] = self.df['values'].apply(lambda x: number_ascii_inseq(x) )     
        print("adding similarity index with a threshold of {th} and {perm} permutations".format(th=self.threshhold, perm=self.permutations) )   
        self.df['Similarity_index_1'] = similarity_index(self.df['values'].to_list(),self.threshhold,self.permutations)
        self.y = self.df['Label']
        self.df['features'] = self.df.apply(lambda x:  [x['non_zero_word_count'],x['asci_count_inseq'],x['Similarity_index_1']],axis=1)
        self.x = self.df['features']
        self.df.drop('values', axis=1, inplace=True)
        return        



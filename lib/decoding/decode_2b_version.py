import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
# Ignore warnings
import warnings
import subprocess
from math import ceil
from ast import literal_eval




import json
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from pathlib import Path

import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import f1_score  
from ast import literal_eval



#Lists all fils in a directory having the provided file_type
def list_files(dir,file_type):    
    """
    dir : directory path
    file_type : file times to list
    ------------------------------
    returns all files of file_type in the provided directory
    """
    r = []                                                                                                            
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):                                                                                          
            for file in files:
                if file.endswith(file_type):
                    r.append(os.path.join(subdir, file))                   
    return r


#List split
def split_strings(x):
    return x.split(' ')

# Calculates Ascii character count
def generate_meta(lst):
    """
    lst : list of strings
    ---------------------
    returns a list of the ASCII count of every string in the input list
    """
    
    r=[]
    for e in lst:
        ascii_count= len([ord(c) for c in e if c != "."])
        r.append(ascii_count)
    return r


def filter_func(e):
    return list(filter(None, e.split(' ')))



def leaf_dirs(dir):
    """
    dir : directory path
    --------------------
    returns all leaf directories inside the provided path
    
    """
    
    max_depth = 0
    leaf_dir = []
    for dirpath, dirnames, filenames in os.walk(dir):
        depth= len(dirpath.split(os.sep))
        if max_depth < depth:
            max_depth= depth
            leaf_dir = [dirpath]
        elif max_depth == depth:
            leaf_dir.append(dirpath)
    return(leaf_dir)    


def list_single_dir_files(dir):
    """
    old implementation of list_files function
    """
    for (dirpath, dirnames, filenames) in os.walk(dir):
        return filenames
        break


def decode_dump(dir):
    """
    dir : folder path
    -----------------
    decodes the .raw files in the provided path and creates .txt files
    
    """
    
    files=list_single_dir_files(dir)
    for file in files:
        command=[]
        if file.endswith('.raw'):
            command=["xxd","-c2",dir+'/'+file,dir+"/"+file[:-3]+"txt"]
            command[2]=command[2].replace('\\', '/')
            command[3]=command[3].replace('\\', '/')
            subprocess.call(command)

            
def del_files(dir,file_type):
    """
    dir : folder path
    files_type : file type
    -----------------
    deletes all the files of the provided type in the provided path
    
    """
    
    files=list_single_dir_files(dir)
    for file in files:
        command=[]
        if file.endswith(file_type):
            command=["rm",'-r',dir+"/"+file]
            subprocess.call(command)
    
# Gets key positions and lengths from a list of json files( per directory)    
def offset(file_list):
    """
    file_list : list containing json files absolute paths
    --------------------------------------------------------
    returns a list containing SSH session keys addresses in res1 for each file in the provided list
            a list containing SSH session keys lengths in res2 for each file in the provided list
    if a key is not existant the respective index will receive None
    
    """
    
    res1=[]
    res2=[]
    for json_file in file_list:
        result=[]
        length=[]
        data=None
        f=open(json_file)
        data=json.load(f)
        heap_start=data['HEAP_START']
        keys=[None,None,None,None,None,None]
        length=[data['KEY_A_LEN'],data['KEY_B_LEN'],data['KEY_C_LEN'],data['KEY_D_LEN'],data['KEY_E_LEN'],data['KEY_F_LEN']]
    
        try:
            if int(data['KEY_A_LEN']) > 0:
                keys[0]=data['KEY_A_ADDR']
            else:
                pass      
        except :
            pass
    
        try:
            if int(data['KEY_B_LEN']) > 0:
                keys[1]=data['KEY_B_ADDR']
            else:
                pass      
        except :
            pass
    
        try:
            if int(data['KEY_C_LEN']) > 0:
                keys[2]=data['KEY_C_ADDR']
            else:
                pass      
        except :
            pass
    
        try:
            if int(data['KEY_D_LEN']) > 0:
                keys[3]=data['KEY_D_ADDR']
            else:
                pass      
        except :
            pass
    
        try:
            if int(data['KEY_E_LEN']) > 0:
                keys[4]=data['KEY_E_ADDR']
            else:
                pass      
        except :
            pass


        try:
            if int(data['KEY_F_LEN']) > 0:
                keys[5]=data['KEY_F_ADDR']
            else:
                pass      
        except :
            pass


        for e in keys:
            if e:
                result.append(hex(int(e, 16)-int(heap_start, 16)))
            else:
                result.append(None)
        res1.append(result)
        res2.append(length)
    return res1, res2    

# Takes in key legnths and start addresses and returns addresses of entire keys
def clean_offset_len(offset,length):
    """
    offset : list containing SSH session keys adresses
    length : list containing SSH session keys lengths 
    ---------------------------------------------------
    returns clean_offsets : address representation not starting with 0x
            key_offsets : same as the input offset
            key_lengths : key length in number of words; original length/ world length
    
    """

    key_offsets=[]
    key_lengths=[0]*len(length)    

    for j in range(len(length)):       
        key_lengths[j] = int(ceil(float(length[j])/2))
                
    for i in range(len(offset)):
        if offset[i]:
            e=offset[i]
            
            for k in range(0,key_lengths[i]):
                key_offsets.append(e)
                e=hex(int(e, 16) + int('2', 16))
    clean_offsets=[None]*len(key_offsets)           
    for i in range(len(key_offsets)):
        if key_offsets[i]:
            temp=key_offsets[i].split('x')
            comp='0'*(8-len(temp[1]))
            clean_offsets[i]=comp+temp[1]
  
    return clean_offsets, key_offsets, key_lengths


def create_labels(offsets,lengths,val):
    """
    offset : list containing SSH session keys adresses
    length : list containing SSH session keys lengths 
    val : contains a list of all addresses in a file
    ---------------------------------------------------
    returns clean_offsets : address representation not starting with 0x
            old_offsets : same as the input offset
            classes : a list of labels 0/1 depending on the existance of keys and matches the provided input of all addresses in a file
    
    """ 
    
    classes=[0]*len(val)
    clean_offsets, old_offsets, key_lengths=clean_offset_len(offsets,lengths)
    for e in clean_offsets:
        if e:
            id=val.index(e)
            classes[id] = 1
    return classes, clean_offsets, old_offsets

# creates a file dataframe
def create_file_df(file_data):
    """
    file_data : list containing [txt_file abs path,json_file abs path,list of SSH key addresses,list of SSH key lengths]
    --------------------------------------------------------------------------------------------------------------------
    returns a dataframe having columns ['offset','hex_values','content','class']
    each row is a line in the input txt_file
                offset : address
                hex_values : hexadecimal characters in the offset
                content : actual data
                class : 1 if the offset contains a key, else 0
    
    """
    columns=['Data']
    df=pd.read_csv(file_data[0],delimiter='\t',names=columns,header=None)
    df['offset']=df['Data'].str[:8]
    df['hex_values']=df['Data'].str[10:]
    df['hex_values']=df['hex_values'].str[:-4]
    df['content']=df['Data'].str[-2:]
    df=df.drop('Data',axis=1)
    df=df[df['offset'] !='*']
    val=df['offset'].tolist()
    labels, key_offsets, old_offsets=create_labels(file_data[2],file_data[3],val)
    df['class']= labels
    df.index = np.arange(0, len(df) )
    df = df.drop(df[(df['class'] == 1) & (df['hex_values'] == '0000')].index)
    df = df.drop(df[(df['content'] == "..") & (df['hex_values'] == '0000')].index)
    df = df.reset_index()
    return df


def create_seq_df(df,N):
    
    """
    df : dataframe created by create_df_file function
    N : sequence size
    --------------------------------------------------------------------------------------------------------------------
    returns a dataframe having columns ['address_range','hex_values','values','classes','ASCII_count','Label']
    each is N word sequence create from df rows
                        address_range : list containing N addresses in sequential order 
                        hex_values : list containing N hex words in sequential order
                        values : list containing N values in sequential order
                        classes : list containing 0 or 1 for each of the N entries
                        ASCII_count : list containing ASCII character count for each of N words
                        Label : integer 1 or 0 = max(classes)
    
    """   
    
    hexa=[]
    content =[]
    classes=[]

    counter=0 
    temp_hex_list=[]
    hex_list=[]

    temp_content_list = []
    content_list=[]

    temp_labels_list=[]
    labels_list=[]

    temp_offset_list=[]
    offset_list=[]

    for index, row in df.iterrows():  
        if (len(hex_list)< len(df)//N):
            
            if (counter<N  ):   
                
                temp_offset_list.append(df.iloc[index,1])
                temp_hex_list.append(df.iloc[index,2])
                temp_content_list.append(df.iloc[index,3])
                temp_labels_list.append(df.iloc[index,4])
                counter=counter+1
            elif ((counter == N)):
                offset_list.append(temp_offset_list)
                hex_list.append(temp_hex_list)
                content_list.append(temp_content_list)
                labels_list.append(temp_labels_list)
                counter=0 
                temp_hex_list=[]
                temp_content_list=[]
                temp_labels_list=[]
                temp_offset_list=[]
                temp_offset_list.append(df.iloc[index,1])
                temp_hex_list.append(df.iloc[index,2])
                temp_content_list.append(df.iloc[index,3])
                temp_labels_list.append(df.iloc[index,4])
                counter=counter+1
        else:
            temp_offset_list.append(df.iloc[index,1])
            temp_hex_list.append(df.iloc[index,2])
            temp_content_list.append(df.iloc[index,3])
            temp_labels_list.append(df.iloc[index,4])
            counter=counter+1

    while (len(temp_hex_list) < N ):
        temp_hex_list.append("00")
        temp_content_list.append('..')
        temp_labels_list.append(0)
        temp_offset_list.append(None)
        offset_list.append(temp_offset_list)
        hex_list.append(temp_hex_list)
        content_list.append(temp_content_list)
        labels_list.append(temp_labels_list)



    file_df = pd.DataFrame({'address_range':offset_list})
    file_df['hex_values'] = hex_list
    file_df['values'] = content_list
    file_df['classes'] = labels_list
    file_df['ASCII_count'] = file_df['values'].apply(lambda x: generate_meta(x) )
    file_df['Label']= file_df['classes'].map(lambda a: max(a))


    return file_df   


def file_save_set(data,file):
    """
    data : dataframe to save as csv
    file : folder path
    """
    filename=file.split('/')
    filename_list1=filename[-3:]
    name='-'.join(filename_list1)       
    name_dir='./2BCSV_Data/'
    sub_dir = filename[0]
    name_file=''+name+'.csv'
    output_dir=Path(name_dir+'/'+sub_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name=name_dir+sub_dir+'/'+name_file
    data.to_csv(name,sep='\t',index=False)
    print('------------------ saved ' +name +' --------------------')
    return



def transform(path,N):
    """
    path: root path for data folder
    N : desiered sequence size
    """
    
    leaf_direc =leaf_dirs(path) 
    for direc in leaf_direc:
        print('######################### Processing', direc,"#############################")
        text_files = list_files(direc, '.txt')   ### a list containing all .txt file names in the leaf directory direc
        json_files = list_files(direc, '.json')  ### a list containing all .json file names in the leaf directory direc
        pos, lengths = offset(json_files)        ### two lists each element contains respectively key addresses and lengths for each file in the leaf directory direc
        it= len(text_files)
        file_dfs=[]
        for i in range(it):
            file_data=[text_files[i],json_files[i],pos[i],lengths[i]]  
            file_df=create_file_df(file_data) ### returns a dataframe for each txt file in direc
            seq_df=create_seq_df(file_df,N)   ### for each file dataframe join N rows into 1 row and returns seq_df 
            file_dfs.append(seq_df)
        folder_df=pd.concat(file_dfs)         ### joining all seq_df into one folder dataframe
        folder_df = folder_df.reset_index()   
        file_save_set(folder_df,str(direc))   ### saving the folder dataframe as csv file
        
    return



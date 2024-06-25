from collections import Counter
import os
import random
from math import ceil

import numpy as np
import pandas as pd
import torch

from torch.utils.data import  Dataset
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import  RandomUnderSampler
from lib.functions import (
    extract_file_names,
    find_pattern,
    get_keys_info,
    sub_value_from_list,
    get_keys_infov2
)


#Malloc header
class DSMDataset(Dataset):
    def __init__(self,path, version, length,  sampling = False, sampling_factor= None):
        super(DSMDataset, self).__init__()
        if length is not None and version is not None:
            verified_path = path + '/' +version+'/'+length
        elif length is None and version is not None:
            verified_path = path + '/' +version
        elif length is None and version is None:
            verified_path = path
        self.path = verified_path
        self.version = version
        self.lenght = length
        self.raw_files = sorted(extract_file_names(self.path, "raw"))
        self.json_files = sorted(extract_file_names(self.path, "json"))
        self.sampling_factor = sampling_factor
        self.sampling = sampling
        print(f'Loading files from {self.path}')
    def __getitem__(self, index):
        raw_path= self.raw_files[index]
        json_path= self.json_files[index]

        key_offsets, lengths = get_keys_infov2(json_path)
        lengths = [int(x) for x in lengths]
        key_offsets = [int(x) for x in key_offsets]
        with open(raw_path, 'rb') as file:
            file_bytes = bytearray(file.read())
            file_data = [element for element in file_bytes]
        
        file_labels = [0]* len(file_data)
        for idx in range(len(key_offsets)):
            file_labels[key_offsets[idx]-1] = 1 # State 1 (malloc header)
            positive_bytes = [num for num in range(key_offsets[idx], key_offsets[idx]+ lengths[idx] )] # State 2 ( key bytes)
            for positive_byte in positive_bytes:
                file_labels[positive_byte]= 2
            file_labels[key_offsets[idx] + max(lengths[idx],16)] = 3 # State 3 (first negative byte after a key)
        file_labels[-1] = 0
        grouped_data = [file_data[i:i+8] for i in range(0, len(file_data), 8)]
        grouped_labels = [file_labels[i:i+8] for i in range(0, len(file_labels), 8)]

        new_positions = []
        modified_lengths= []

        length_val = 0
        key_pos= []
        for idx, sublist in enumerate(grouped_labels):
            if 2 in sublist:
                length_val+= 1
                key_pos.append(idx)
            else:
                if len(key_pos)>0:
                    new_positions.append(key_pos)
                    modified_lengths.append(length_val)
                length_val = 0
                key_pos= []
                

        pruned_labels = []
        for label in grouped_labels:
            if 1 in label:
                pruned_labels.append(1)
            elif 3 in label:
                pruned_labels.append(3)
            elif 2 in label:
                pruned_labels.append(2)
            else:
                pruned_labels.append(0)

        if self.sampling:
            sampled_data, sampled_labels = self.sample_data(data= grouped_data,labels= pruned_labels, modified_positions= new_positions)
            return sampled_data, sampled_labels
        return grouped_data, pruned_labels
        
    def __len__(self):
        return len(self.raw_files)

    def sample_data(self, data, labels, modified_positions):
        processed_data = data
        processed_labels = labels
        # Adding next segment or malloc header position
        new_positions= modified_positions
        for index in range(len(modified_positions)):
            new_positions[index] = [modified_positions[index][0] - 1]+modified_positions[index] + [modified_positions[index][-1] + 1]
        #Extracting keys
        keys_to_sample = []
        labels_to_sample = []
        for index_list in new_positions:
            key_sublist = [data[index] for index in index_list]
            label_sublist = [labels[index] for index in index_list]
            keys_to_sample.append(key_sublist)
            labels_to_sample.append(label_sublist)
        #Sampling
        label_distribution= Counter(labels)
        factor = int((label_distribution[0]/ label_distribution[1])//self.sampling_factor) ####
        for _ in range(factor):
            chosen_key_index = random.randint(0, len(keys_to_sample) - 1)
            chosen_key = keys_to_sample[chosen_key_index]
            chosen_label = labels_to_sample[chosen_key_index]

            available_indices = [i for i, val in enumerate(processed_labels) if val == 0]
            if not available_indices:
                break  
            index_to_insert = random.choice(available_indices)
            processed_data = processed_data[:index_to_insert] + chosen_key + processed_data[index_to_insert:]
            processed_labels = processed_labels[:index_to_insert] + chosen_label + processed_labels[index_to_insert:]
        return processed_data, processed_labels

class UnlabeledDSMDataset(Dataset):
    def __init__(self,path, version, length):
        super(UnlabeledDSMDataset, self).__init__()
        if length is not None and version is not None:
            verified_path = path + '/' +version+'/'+length
        elif length is None and version is not None:
            verified_path = path + '/' +version
        elif length is None and version is None:
            verified_path = path
        self.path = verified_path
        self.version = version
        self.lenght = length
        self.raw_files = sorted(extract_file_names(self.path, "raw"))
        print(f'Loading files from {self.path}')
    def __getitem__(self, index):
        raw_path= self.raw_files[index]

        with open(raw_path, 'rb') as file:
            file_bytes = bytearray(file.read())
            file_data = [element for element in file_bytes]
        grouped_data = [file_data[i:i+8] for i in range(0, len(file_data), 8)]
        return grouped_data
        
    def __len__(self):
        return len(self.raw_files)

class ByteDataset(Dataset):
    def __init__(self,path, window_size, stride, enc_length= None, version= None):
        super(ByteDataset, self).__init__()
        self.path = path
        self.files = []
        self.nbr_keys_per_file = []
        self.byte_array = bytearray()
        self.patterns = []
        self.data = []
        self.offsets = []
        self.lengths = []
        self.pattern_labels = []
        self.window_size = window_size
        self.stride = stride
        self.encryption_length = enc_length
        self.version = version
        
        self.decode_heap()
        self.offsets = sub_value_from_list(self.offsets, 8)

        self.byte_array_to_int()
        self.get_potential_header()

    def decode_heap(self):
        previous_len = 0
        for file_path in self.path:
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
                self.files.append(file_path)
                self.byte_array += file_bytes
                key_offsets, lengths = get_keys_info(file_path)
                self.lengths.extend(lengths)
                self.nbr_keys_per_file.append(len(key_offsets))
                new_key_indexes = []
                new_key_indexes = [key_offset + previous_len for key_offset in key_offsets]
                previous_len += len(file_bytes)
                self.offsets.extend(new_key_indexes)

    def byte_array_to_int(self):
        temp = [element for element in self.byte_array]
        self.data.extend(list(temp))

    def get_potential_header(self):
        for pos in self.offsets:
            header_arr= []
            header_arr = [byte for byte in self.data[pos:pos+8]]
            self.patterns.append(header_arr)
        self.pattern_labels = [list(x) for x in list(set(tuple(x) for x in self.patterns))]

    def __getitem__(self, index):
        # Get a window of data and its label
        while True:
            position_inWindow = np.inf
            pattern_inWindow = np.inf
            length_inWindow = np.inf
            if index == 0:
                start_idx = 0
            else:
                start_idx = index * (self.window_size - self.stride)
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]
            window_label = 0
            for patt in self.pattern_labels:   # [65,0,0,0,0,0,0] [49,0,0,0,0,0,0]
                if all(x in window_data for x in patt) and any((start_idx< offset< end_idx) and( find_pattern(window_data, patt, offset-start_idx)) for offset in self.offsets) :
                    window_label = 1
                    position_inWindow = [offset - start_idx for offset in self.offsets if start_idx < offset < end_idx][0]
                    pattern_inWindow = torch.tensor(window_data[position_inWindow:position_inWindow+8], dtype=torch.float32).unsqueeze(1)
                    offset_pos = position_inWindow + start_idx
                    offset_idx = self.offsets.index(offset_pos)
                    length_inWindow = self.lengths[offset_idx]
                    if position_inWindow < self.window_size-self.stride:
                        continue
            if len(window_data)< self.window_size:
                break
            else:
                index = index+1
            data_tensor = torch.tensor(window_data, dtype=torch.float, requires_grad= True)
            label_tensor = torch.tensor(window_label, dtype=torch.float, requires_grad=True)
            one_hot_labels = torch.nn.functional.one_hot(label_tensor.long(), num_classes=2).squeeze(0)
            return data_tensor, one_hot_labels, position_inWindow, pattern_inWindow, length_inWindow

    def __len__(self):
        # Return the number of windows
        if self.window_size-self.stride != 0:
            return  ceil((len(self.data) - self.window_size) / (self.window_size - self.stride)) 
        else:
            raise Exception('the window_size cannot be equal to the stride size')

    def describe(self):
        # Count the number of elements in each class and print the counts
        counts = {"1": 0, "0": 0}
        for i in range(len(self)):
            data, label, position, pattern, length= self.__getitem__(i)  #type: ignore
            label_class = label.argmax(dim=0).item()  # Find the index of the maximum value in the label tensor
            if label_class == 1:
                counts["1"] += 1
            else:
                counts["0"] += 1
        print(f"number of samples: {len(self)}")
        print("Class Counts:")
        for label, count in counts.items():
            print(f"{label}: {count}")

class PositiveClassByteDataset(ByteDataset):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.positive_samples = []
        self.positions = []
        self.length = []
        self.pattern = []
        self.filter_positive_samples()

    def filter_positive_samples(self):
        for i in range(super().__len__()):
            data, label, position, pattern, length = super().__getitem__(i)  #type: ignore
            label_class = label.argmax(dim=0).item()
            
            if label_class == 1 :  #and position < self.window_size-self.stride
                self.positive_samples.append(data)
                self.positions.append(position)
                self.pattern.append(pattern)
                self.length.append(length)
            else:
                continue
 
    def __getitem__(self, index):
        if index >=  self.__len__():
            raise IndexError("Index out of range")

        data = self.positive_samples[index]
        position = self.positions[index]
        pattern = self.pattern[index]
        length = self.length[index]
        int_label = torch.tensor(1)
        label = torch.nn.functional.one_hot(int_label, num_classes=2).squeeze(0)
        return data, label,position, pattern, length
        
    def __len__(self):
        return len(self.positive_samples)

class Unlabeled_ByteDataset(Dataset):
    def __init__(self,path, window_size, stride, enc_length, version):
        super(Unlabeled_ByteDataset, self).__init__()
        self.path = path
        self.files = []
        self.byte_array = bytearray()
        self.data = []
        self.window_size = window_size
        self.stride = stride
        self.encryption_length = enc_length
        self.version = version
        
        self.decode_heap()
        self.byte_array_to_int()

    def decode_heap(self):
        previous_len = 0
        for file_path in self.path:
            with open(file_path, 'rb') as file:
                file_bytes = file.read()
                self.files.append(file_path)
                self.byte_array += file_bytes
                previous_len += len(file_bytes)  

    def byte_array_to_int(self):
        temp = [element for element in self.byte_array]
        self.data.extend(list(temp))
    
    def __getitem__(self, index):
        while True:
            if index == 0:
                start_idx = 0
            else:
                start_idx = index * (self.window_size - self.stride)
            end_idx = start_idx + self.window_size
            window_data = self.data[start_idx:end_idx]
            if len(window_data)< self.window_size:
                break
            else:
                index = index+1
            data_tensor = torch.tensor(window_data, dtype=torch.float, requires_grad= True)
            return data_tensor 
        
    def __len__(self):
        # Return the number of windows
        if self.window_size-self.stride != 0:
            return  ceil((len(self.data) - self.window_size) / (self.window_size - self.stride)) 
        else:
            raise Exception('the window_size cannot be equal to the stride size')

class data_set(Dataset):
    def __init__(self,data,task,majority_rate= None,sample = None ):
        if task not in ['regression','classification']:
            raise ValueError("task must be regression or classification")
        if (majority_rate and not sample) or (sample and not majority_rate):
            raise ValueError("Must specify sampeling type and sampleing rate or none")
        if sample and sample not in ['under','over',"balanced"]:
            raise ValueError("sample must be under, over, balanced or None")
        if task =='regression' and sample in ['over','balanced']:
            raise ValueError('For regression only undersampeling is suppored (under for undersampeling)')
        self.data=data.df
        self.seq_len = data.seq_len
        self.x=data.df['hex_values']
        self.y=None
        self.classes = None
        self.task=task


        if self.task == 'regression':
            self.y=data.df['Position'] 
            self.get_classes()
            
            if sample == 'under':
                self.sample_rate=int(self.classes[self.seq_len]*(1-majority_rate))
                self.reg_under_sampeling() 
            self.get_classes()


        elif self.task == 'classification':
            self.sample_rate=majority_rate
            self.y=data.df['Label']

            if sample == 'under':
                self.class_under_sampeling()
            elif sample== 'balanced':
                self.class_sampeling()
            elif sample == 'over':
                self.class_over_sampeling()
            self.get_classes()

        self.vocab=dict()
        self.build_vocab()
        self.vectorize()
        print("####  Done ####")




    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def __setitem__(self,idx,value):
        self.x[idx]=value[0]
        self.y[idx]=value[1]


    def to_torch(self):
        self.x=torch.LongTensor(self.x)
        self.y=torch.FloatTensor(self.y)


    def stats(self):
        for key, value in self.classes.items():
            if self.task == 'classification':
                print('Class {} count : {} '.format(key, value))
            elif self.task == 'regression':
                print('Position {} count : {} '.format(key, value))

    
          
    def build_vocab(self):
        self.vocab={}
        print('#### Building vocab ####')     
        i=1
        for value in self.x:
            for element in value:
                if element in self.vocab:
                    pass
                else:
                    self.vocab[element]=i
                    i=i+1
            
    def vectorize(self):
        print('#### vectorization ####')
        for value in self.x:
            for i in range(len(value)):
                try:
                    value[i]=self.vocab[value[i]]
                except:
                    value[i]=0  

    def class_over_sampeling(self):  
        print('#### oversampleing ####: ') 
        self.x=self.x.tolist()
        self.y=self.y.tolist()   
        oversample = SMOTE()
        self.x , self.y=oversample.fit_resample(self.x, self.y)
        self.x=pd.Series(self.x)
        self.y=pd.Series(self.y)

        
    def class_sampeling(self):
        print('#### balanced sampeling ####: ')
        self.x=self.x.tolist()
        self.y=self.y.tolist()  
        over = SMOTE(sampling_strategy=1-self.sample_rate)
        under = RandomUnderSampler(sampling_strategy=self.sample_rate)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        self.x , self.y=pipeline.fit_resample(self.x, self.y)
        self.x=pd.Series(self.x)
        self.y=pd.Series(self.y)

        
    def class_under_sampeling(self):
        print('#### undersampeling ####: ')
        self.x=self.x.tolist()
        self.y=self.y.tolist()  
        under = RandomUnderSampler(random_state=42,sampling_strategy=self.sample_rate)
        self.x , self.y=under.fit_resample(self.x, self.y)
        self.x=pd.Series(self.x)
        self.y=pd.Series(self.y)
  
        
    def reg_under_sampeling(self) :
        print('#### undersampeling ####: ')
        self.x=self.x.tolist()
        self.y=self.y.tolist()  
        under = RandomUnderSampler(random_state=42,sampling_strategy={self.seq_len : self.sample_rate},replacement=True)
        self.x , self.y=under.fit_resample(self.x, self.y)
        self.x=pd.Series(self.x)
        self.y=pd.Series(self.y)
  
    def get_classes(self):
        classes = self.y.unique()
        class_dict= dict(zip(classes, [0]*len(classes)))
        for value in self.y:
            class_dict[value] = class_dict[value]+1
        self.classes = class_dict

              
# No malloc header
class DSMDataset_old(Dataset):
    def __init__(self,path, version, length,  sampling = False, sampling_factor= None):
        super(DSMDataset_old, self).__init__()
        self.path = path + '/' +version+'/'+length
        self.version = version
        self.lenght = length
        self.raw_files = sorted(extract_file_names(self.path, "raw"))
        self.json_files = sorted(extract_file_names(self.path, "json"))
        self.sampling_factor = sampling_factor
        self.sampling = sampling
    def __getitem__(self, index):
        raw_file= self.raw_files[index]
        json_file= self.json_files[index]

        raw_path= self.path+'/'+raw_file
        json_path = self.path+'/'+json_file


        key_offsets, lengths = get_keys_infov2(json_path)
        lengths = [int(x) for x in lengths]
        key_offsets = [int(x) for x in key_offsets]
        with open(raw_path, 'rb') as file:
            file_bytes = bytearray(file.read())
            file_data = [element for element in file_bytes]
        
        file_labels = [0]* len(file_data)
        for idx in range(len(key_offsets)):
            file_labels[key_offsets[idx]] = 1 # State 1 (first key byte)
            positive_bytes = [num for num in range(key_offsets[idx]+1, key_offsets[idx]+ lengths[idx] )] # State 2 (remaining key bytes)
            for positive_byte in positive_bytes:
                file_labels[positive_byte]= 2
            file_labels[key_offsets[idx] + max(lengths[idx],16)] = 3 # State 3 (first negative byte after a key)
        file_labels[-1] = 0
        grouped_data = [file_data[i:i+8] for i in range(0, len(file_data), 8)]
        grouped_labels = [file_labels[i:i+8] for i in range(0, len(file_labels), 8)]
        new_positions = []
        for idx, sublist in enumerate(grouped_labels):
            if any(value == 2 for value in sublist):
                new_positions.append(idx)
        pruned_labels = []
        for label in grouped_labels:
            if 1 in label:
                pruned_labels.append(1)
            elif 3 in label:
                pruned_labels.append(3)
            elif 2 in label:
                pruned_labels.append(2)
            else:
                pruned_labels.append(0)


        if self.sampling:
            sampled_data, sampled_labels = self.sample_data(grouped_data, pruned_labels, new_positions, lengths)
            return sampled_data, sampled_labels
        return grouped_data, pruned_labels
        
    def __len__(self):
        return len(self.raw_files)

    def sample_data(self, data, labels, positions, lengths):
        processed_data = data
        processed_labels = labels
        #Grouping lengths
        modified_lengths = [length // 8 if length % 8 == 0 else length // 8 + 1 for length in lengths]

        modified_positions = []
        start_idx = 0
        #Grouping positions
        for group_size in modified_lengths:
            group = positions[start_idx:start_idx + group_size]
            modified_positions.append(group)
            start_idx += group_size
        # print('mod_lengths', modified_lengths)
        # print('mod_pos', modified_positions)
        # Adding next segment or malloc header position
        for index in range(len(modified_positions)):
            modified_positions[index] = modified_positions[index] + [modified_positions[index][-1] + 1]
        # print('mod_pos', modified_positions)
        #Extracting keys
        keys_to_sample = []
        labels_to_sample = []
        for index_list in modified_positions:
            key_sublist = [data[index] for index in index_list]
            label_sublist = [labels[index] for index in index_list]
            keys_to_sample.append(key_sublist)
            labels_to_sample.append(label_sublist)
        # print('key to sample', keys_to_sample)
        # print('label to sample', labels_to_sample)
        #Sampling
        label_distribution= Counter(labels)
        factor = int((label_distribution[0]/ label_distribution[1])//self.sampling_factor) ####
        for _ in range(factor):
            chosen_key_index = random.randint(0, len(keys_to_sample) - 1)
            chosen_key = keys_to_sample[chosen_key_index]
            chosen_label = labels_to_sample[chosen_key_index]

            available_indices = [i for i, val in enumerate(processed_labels) if val == 0]
            if not available_indices:
                break  
            index_to_insert = random.choice(available_indices)
            processed_data = processed_data[:index_to_insert] + chosen_key + processed_data[index_to_insert:]
            processed_labels = processed_labels[:index_to_insert] + chosen_label + processed_labels[index_to_insert:]
        return processed_data, processed_labels
        






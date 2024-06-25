import numpy as np
import json
from tqdm.notebook import tqdm
import os
import zipfile
import shutil
import subprocess
from lib.decoding.decode_8b_version import transform, del_files
import torch
from torch.utils.data import WeightedRandomSampler
from sklearn.utils import class_weight
import pandas as pd
from ast import literal_eval
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)
import datetime
from datasketch import MinHash, MinHashLSH


def print_timestamp():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(timestamp)

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
        
def number_non_zero(lst):
    res=0
    for word in lst:
        if word != '........':
            res = res +1
    return res


def number_ascii_inseq(lst):
    res=0
    for word in lst:
        res = res + len([c for c in word if c != "."])
    return res

def similarity_index(lst,precision,permutation):
    result=[]
    lsh = MinHashLSH(threshold=precision, num_perm=permutation)
    minhashes = {}
    for idx, val in enumerate(lst):
        minhash = MinHash(num_perm=permutation)
        for d in val:
            minhash.update(d.encode('utf-8'))   
        lsh.insert(idx, minhash)
        minhashes[idx] = minhash
    for i in range(len(minhashes.keys())):
        result.append(len(lsh.query((minhashes[i]))))
    return result
    

def model_size(model):  
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.2f}MB'.format(size_all_mb))
    
def lit_eval_csv_read(path):
    df = pd.read_csv(path,sep='\t')
    df['address_range'] = df['address_range'].apply(literal_eval)
    df['hex_values'] = df['hex_values'].apply(literal_eval)
    df['values'] = df['values'].apply(literal_eval)
    df['ASCII_count'] = df['ASCII_count'].apply(literal_eval)
    df['classes'] = df['classes'].apply(literal_eval)
    return df
        

def get_metrics(y_test, y_predict):
    accuracy= accuracy_score(y_test,y_predict)
    report = classification_report(y_predict, y_test)
    cm = confusion_matrix(y_test, y_predict)
    cm_display = ConfusionMatrixDisplay(cm)
    fpr, tpr, _ = roc_curve(y_test, y_predict, pos_label=1)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    print(f' Accuracy : {accuracy} \n report : \n{report}  ')
    print(cm)
    cm_display.plot()
    roc_display.plot()


def get_dataset_file_paths(path, deploy=False):
    import glob
    import os
    paths = []

    file_paths = []
    
    # Get all subdirectories 
    sub_dir = os.walk(path)
    for directory in sub_dir:
        paths.append(directory[0])
    
    # Get all unique directory paths
    paths = set(paths)
    
    for path in paths:
        
        files = glob.glob(os.path.join(path, '*.raw'), recursive=False)
        
        # If there are no heap files in the current directory
        if len(files) == 0:
            continue

        for file in files:
            
            # Get the JSON file corresponding to the raw heap file
            key_file = file[:-9] + ".json"
            
            # Compute transitions for all heap files that have a corresponding key file
            if os.path.exists(key_file):
                file_paths.append(file)

            else:
                print("Corresponding Key file does not exist for :%s" % file)

    return file_paths

def get_json_file(file_path):
    return file_path[:-9]+".json"

def sub_value_from_list(lst, value):
        return [x - value for x in lst if x]

def get_keys_info(file_path):

    json_file=get_json_file(file_path)
    with open(json_file) as file:
        data = json.load(file)
    heap_start=data['HEAP_START']
    keys=[None,None,None,None,None,None]
    length=[data['KEY_A_LEN'],data['KEY_B_LEN'],data['KEY_C_LEN'],data['KEY_D_LEN'],data['KEY_E_LEN'],data['KEY_F_LEN']]
    adresses=[]
    try:
        if int(data['KEY_A_LEN']) > 0:
            keys[0]=data['KEY_A_ADDR']     
    except :
        pass

    try:
        if int(data['KEY_B_LEN']) > 0:
            keys[1]=data['KEY_B_ADDR']     
    except :
        pass

    try:
        if int(data['KEY_C_LEN']) > 0:
            keys[2]=data['KEY_C_ADDR']   
    except :
        pass

    try:
        if int(data['KEY_D_LEN']) > 0:
            keys[3]=data['KEY_D_ADDR']  
    except :
        pass

    try:
        if int(data['KEY_E_LEN']) > 0:
            keys[4]=data['KEY_E_ADDR']   
    except :
        pass

    try:
        if int(data['KEY_F_LEN']) > 0:
            keys[5]=data['KEY_F_ADDR']     
    except :
        pass
    for e in keys:
        if e:
            res=(int(e, 16)-int(heap_start, 16))
            adresses.append(res)
        else:
            adresses.append(None) 
        adresses = list(filter(lambda x: x is not None, adresses))
        length = list(filter(lambda x: x != '0', length))
    return adresses, length


def construct_transition_matrix(file_paths):
    
    # Create an empty transition matrix
    transition_matrix = np.zeros(shape=(256,256))
    
    for file_path in tqdm(file_paths, desc='Reading Heap'):
        with open(file_path, 'rb') as fp:
            heap = bytearray(fp.read())
        
        for idx in range(len(heap)-1):
            curr_val = heap[idx]
            next_val = heap[idx + 1]
            transition_matrix[curr_val][next_val] += 1
    return transition_matrix



def extract_keys(path):
    keys_offset, lengths = get_keys_info(path)
    with open(path, 'rb') as fp:
        heap = bytearray(fp.read())
    hex_keys=[]
    bin_keys=[]
    for idx in range(len(keys_offset)):
        if keys_offset[idx]:
            key_val = heap[keys_offset[idx]:keys_offset[idx]+int(lengths[idx])].hex()
            hex_keys.append(key_val)
            bin_keys.append(bin(int(key_val, base=16))[2:])
        else:
            hex_keys.append(None)
            bin_keys.append(None)
    return hex_keys, bin_keys


def byte_to_bit(byte):
    return bin(int(byte.hex()), base=16)[2:]


def extract(directory= 'Data'):
    zip_list= ['Validation.zip?download=1', 'Training.zip?download=1']
    
    for zip_filename in zip_list:
        output_dir = f"./{directory}"  # Set the desired output directory here

        # Unzip the file
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)

        # Optional: Delete the zip file after extraction
        os.remove(zip_filename)

def delete_old_versions(version_folders_to_delete, directory= 'Data'):
    

    data_path = f"./{directory}"

    # Ensure the data path is a valid directory
    if not os.path.isdir(data_path):
        raise ValueError("data_path should be a valid directory.")

    # Iterate through all leaf directories under "./Data"
    for root, dirs, files in os.walk(data_path):
        # Check if the current directory's name matches any of the version folders to delete
        if os.path.basename(root) in version_folders_to_delete:
            print("Deleting:", root)
            shutil.rmtree(root)
    return

def data_split(directory= 'Data'):
    data_path = f"./{directory}"
    training_folder = os.path.join(data_path, "Training")
    validation_folder = os.path.join(data_path, "Validation")
    performance_test_folder = os.path.join(data_path, "Performance Test")

    # Ensure the data path is a valid directory
    if not os.path.isdir(data_path):
        raise ValueError("data_path should be a valid directory.")

    # Check if the "Validation" folder exists before renaming
    if os.path.exists(validation_folder):
        # Check if "Performance Test" folder already exists, and if so, remove it before renaming
        if os.path.exists(performance_test_folder):
            print("Deleting existing 'Performance Test' folder...")
            shutil.rmtree(performance_test_folder)

        # Rename the "Validation" folder to "Performance Test"
        print("Renaming 'Validation' folder to 'Performance Test'...")
        os.rename(validation_folder, performance_test_folder)
        print("Rename complete.")

        # Copy the content of "Training" to a new "Validation" folder
        print("Splitting the training data 80% Training and 20% Validation'...")
        shutil.copytree(training_folder, validation_folder)
        delete_first_part(directory)
        delete_second_part(directory)
        print("Split complete.")
    else:
        print("The 'Validation' folder does not exist.")

    return



def delete_first_part(directory= 'Data'):
    validation_path = get_leaf_folders(f'./{directory}/Validation')
    for folder_path in validation_path:
        file_pairs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_pairs.append(file_name)

        file_pairs.sort()  # Sort the file pairs in alphabetical order

        split_len = int(len(file_pairs) * 0.8)
        files_to_delete = file_pairs[:split_len]

        for file_name in files_to_delete:
            json_file = os.path.join(folder_path, file_name)
            raw_file = os.path.join(folder_path, file_name[:-5] + "-heap.raw")  # Assuming the raw file has the same name except for the extension
            os.remove(json_file)
            os.remove(raw_file)

    return

def delete_second_part(directory= 'Data'):
    validation_path = get_leaf_folders(f'./{directory}/Training')
    for folder_path in validation_path:
        file_pairs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".json"):
                file_pairs.append(file_name)

        file_pairs.sort()  # Sort the file pairs in alphabetical order

        split_len = int(len(file_pairs) * 0.8)
        files_to_delete = file_pairs[split_len:]

        for file_name in files_to_delete:
            json_file = os.path.join(folder_path, file_name)
            raw_file = os.path.join(folder_path, file_name[:-5] + "-heap.raw")  # Assuming the raw file has the same name except for the extension
            os.remove(json_file)
            os.remove(raw_file)
    return

def get_leaf_folders(dir):
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
    for (dirpath, dirnames, filenames) in os.walk(dir):
        return filenames
        break
        

def decode_dump(dir):
    files=list_single_dir_files(dir)
    for file in files:
        command=[]
        if file.endswith('.raw'):
            command=["xxd","-c8","-a",dir+'/'+file,dir+"/"+file[:-3]+"txt"]
            command[3]=command[3].replace('\\', '/')
            command[4]=command[4].replace('\\', '/')
            subprocess.call(command)

def full_decode(directory= 'Data'):
    path = f"./{directory}"
    leaf_folders = get_leaf_folders(path)
    print(f'Decoding all .raw files into .txt files from : {leaf_folders}')

    for folder in leaf_folders:
        decode_dump(folder)

def full_txt_delete(directory= 'Data'):
    path = f"./{directory}"
    leaf_folders = get_leaf_folders(path)
    print('Deleteing all previously generated .txt files')
    for folder in leaf_folders:
        del_files(folder,'.txt')

def create_csv(segment_size):
    train_path = './Data/Training'
    validation_path = './Data/Validation'
    test_path = './Data/Performance Test'
    print('Creating CSV files for all the data version in the Data folder')
    transform(train_path, segment_size)
    transform(validation_path, segment_size)
    transform(test_path, segment_size)
    print('Done.')

    
def extract_file_names(folder_path, file_extension):
    files_names = []
    
    # Traverse through all folders and subfolders recursively
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(f'.{file_extension}'):
                file_path = os.path.join(root, file)
                files_names.append(file_path)

    return files_names


def get_keys_infov2(json_file):

    with open(json_file) as file:
        data = json.load(file)
    heap_start=data['HEAP_START']
    keys=[None,None,None,None,None,None]
    length=[data['KEY_A_LEN'],data['KEY_B_LEN'],data['KEY_C_LEN'],data['KEY_D_LEN'],data['KEY_E_LEN'],data['KEY_F_LEN']]
    adresses=[]
    try:
        if int(data['KEY_A_LEN']) > 0:
            keys[0]=data['KEY_A_ADDR']     
    except :
        pass

    try:
        if int(data['KEY_B_LEN']) > 0:
            keys[1]=data['KEY_B_ADDR']     
    except :
        pass

    try:
        if int(data['KEY_C_LEN']) > 0:
            keys[2]=data['KEY_C_ADDR']   
    except :
        pass

    try:
        if int(data['KEY_D_LEN']) > 0:
            keys[3]=data['KEY_D_ADDR']  
    except :
        pass

    try:
        if int(data['KEY_E_LEN']) > 0:
            keys[4]=data['KEY_E_ADDR']   
    except :
        pass

    try:
        if int(data['KEY_F_LEN']) > 0:
            keys[5]=data['KEY_F_ADDR']     
    except :
        pass
    for e in keys:
        if e:
            res=(int(e, 16)-int(heap_start, 16))
            adresses.append(res)
        else:
            adresses.append(None) 
        adresses = list(filter(lambda x: x is not None, adresses))
        length = list(filter(lambda x: x != '0', length))
    return adresses, length


def find_pattern(lst, pattern, given_index):
    pattern_length = len(pattern)
    for i in range(len(lst) - pattern_length + 1):
        if lst[i:i+pattern_length] == pattern:
            if i == given_index:
                return True
    return False

def custom_collate_fn(batch):
    samples, labels, positions, patterns , length= zip(*batch)
    return torch.stack(samples), torch.stack(labels), positions, patterns, length

def get_sampler(dataset):
    labels = []
    for i in range(len(dataset)):
        _, label,_,_,_ = dataset[i]  
        labels.append(label)
    labels = torch.stack(labels)
    class_indices = np.argmax(labels, axis=1).tolist()
    unique_classes = np.unique(class_indices)
    class_weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=class_indices)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    weights = class_weights[class_indices]
    #class_weights = torch.tensor(class_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

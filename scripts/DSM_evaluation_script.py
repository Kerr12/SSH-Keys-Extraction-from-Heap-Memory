from lib.custom_datasets import DSMDataset, UnlabeledDSMDataset
from lib.DSM import DSM
from lib.functions import print_timestamp

print_timestamp()
trial = 20
sampling_factor = 2
embedding_dim = 128
agent_hidden_dim= 256
lr = 0.001
data_version='V_8_8_P1'
length= None
print(f'Deep State Machine Trained on {data_version}/{length}')

print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Validation on {data_version}/64')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

validation_path = 'Data_V2/Validation/basic'
dataset = DSMDataset(validation_path, data_version, length= '64', sampling= False)
print(f"Dataset contains {len(dataset)} files")
print('------------------------------------------------------------------------------------------------------------------------------------------------')
dsm = DSM(embedding_dim=embedding_dim, agent_hidden_state= agent_hidden_dim, cuda_index=3)
dsm.load_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
print_timestamp()
predictions=dsm.infer(dataset= dataset, labeled_dataset=True, suppress_output= False, evaluation_approach='per_file')
print_timestamp() 




print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Validation on {data_version}/32')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

validation_path = 'Data_V2/Validation/basic'
dataset = DSMDataset(validation_path, data_version, length= '32', sampling= False)
print(f"Dataset contains {len(dataset)} files")
print('------------------------------------------------------------------------------------------------------------------------------------------------')
dsm = DSM(embedding_dim=embedding_dim, agent_hidden_state= agent_hidden_dim, cuda_index=3)
dsm.load_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
print_timestamp()
predictions=dsm.infer(dataset= dataset, labeled_dataset=True, suppress_output= False, evaluation_approach='per_file')
print_timestamp() 



print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Validation on {data_version}/24')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

validation_path = 'Data_V2/Validation/basic'
dataset = DSMDataset(validation_path, data_version, length= '24', sampling= False)
print(f"Dataset contains {len(dataset)} files")
print('------------------------------------------------------------------------------------------------------------------------------------------------')
dsm = DSM(embedding_dim=embedding_dim, agent_hidden_state= agent_hidden_dim, cuda_index=3)
dsm.load_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
print_timestamp()
predictions=dsm.infer(dataset= dataset, labeled_dataset=True, suppress_output= False, evaluation_approach='per_file')
print_timestamp() 




print('------------------------------------------------------------------------------------------------------------------------------------------------')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

print(f'Validation on {data_version}/16')
print('------------------------------------------------------------------------------------------------------------------------------------------------')

validation_path = 'Data_V2/Validation/basic'
dataset = DSMDataset(validation_path, data_version, length= '16', sampling= False)
print(f"Dataset contains {len(dataset)} files")
print('------------------------------------------------------------------------------------------------------------------------------------------------')
dsm = DSM(embedding_dim=embedding_dim, agent_hidden_state= agent_hidden_dim, cuda_index=3)
dsm.load_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial)
print('------------------------------------------------------------------------------------------------------------------------------------------------')
print_timestamp()
predictions=dsm.infer(dataset= dataset, labeled_dataset=True, suppress_output= False, evaluation_approach='per_file')
print_timestamp() 
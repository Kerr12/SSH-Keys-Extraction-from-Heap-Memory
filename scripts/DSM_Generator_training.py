from lib.custom_datasets import DSMDataset, UnlabeledDSMDataset
from lib.DSM import DSM
from lib.functions import print_timestamp
print_timestamp()

trial = 8
sampling_factor = 2
embedding_dim = 128
agent_hidden_dim= 256
lr = 0.001

data_version='V_8_8_P1'
length= '64'
print(f"{data_version}/{length} ")

training_path= 'Data_V2/Training/basic'
training_set = DSMDataset(training_path, data_version, length, sampling= True, sampling_factor = sampling_factor)

dsm = DSM(embedding_dim=embedding_dim, agent_hidden_state= agent_hidden_dim)
dsm.load_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial)

print(f"Training dataset contains {len(training_set)} files")
print_timestamp()
generated_data, predicted_states=dsm.train_generator(dataset=training_set, lr= 0.001, suppress_output= False)
dsm.save_model(directory_path= '/root/thesis/dsm_models', data_version= data_version, length= length, trial= trial, type='generation')
print_timestamp()



